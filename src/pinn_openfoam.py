#!/usr/bin/env python3
"""
Minimal steady PINN for OpenFOAM motorBike (simpleFoam) data.
- Loads VTK volume file exported with `foamToVTK`
- Tiny MLP: (x,y,z) -> (u,v,w,p)
- Physics loss: steady incompressible NS (continuity + momentum)
- Optional data loss on (U, p) from VTK

Run with:
  python3 motorbike_pinn.py --vtk path/to/motorBikeSteady_500.vtk

Requirements: pip install pyvista torch numpy
"""
from __future__ import annotations
import argparse
import numpy as np
import pyvista as pv
import torch
import torch.nn as nn
from typing import Tuple

# -------------------- USER CONSTANTS --------------------
EPOCHS     = 800                # was 100
BATCH_COLL = 150_000            # was 5_000
BATCH_DATA = 80_000             # was 10_000
LR         = 5e-4
DEVICE     = "cuda"

# Physical / scaling constants
U_INF  = 30.0    # m/s, free-stream
L_REF  = 1.0     # m, characteristic length
RHO    = 1.225   # kg/m^3
RE     = 2_000_000.0  # Reynolds number -> nu_eff = 1/RE

# -------------------- Autograd helpers --------------------
def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs, inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

def laplacian(scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g = grad(scalar, x)
    parts = []
    for i in range(x.shape[1]):
        dgi_dxi = torch.autograd.grad(
            g[:, i], x, grad_outputs=torch.ones_like(g[:, i]),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0][:, i:i+1]
        parts.append(dgi_dxi)
    return sum(parts)

def vector_laplacian(vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([laplacian(vec[:, i:i+1], x) for i in range(3)], dim=1)

def divergence(vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    comps = [grad(vec[:, i:i+1], x)[:, i:i+1] for i in range(3)]
    return sum(comps)

def convection(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # (u · ∇) u
    J_cols = [grad(u[:, i:i+1], x) for i in range(3)]  # (N,3) each
    J = torch.stack(J_cols, dim=1)                      # (N,3,3)
    ucol = u.unsqueeze(-1)                              # (N,3,1)
    return torch.matmul(J, ucol).squeeze(-1)            # (N,3)

# -------------------- Model --------------------
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, width=128, depth=6, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------- Physics residuals --------------------
def ns_residuals(model: nn.Module, x: torch.Tensor, rho: float = 1.0, nu_eff: float | torch.Tensor = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    x.requires_grad_(True)
    out = model(x)
    u = out[:, 0:3]
    p = out[:, 3:4]

    r_cont = divergence(u, x)                  # ∇·u
    conv = convection(u, x)                    # (u·∇)u
    grad_p = grad(p, x) / rho                  # ∇p/ρ
    lap_u = vector_laplacian(u, x)             # ∇²u

    mom = conv + grad_p - nu_eff * lap_u
    return r_cont, mom

# -------------------- Data loading --------------------
def load_vtk(volume_path: str):
    grid = pv.read(volume_path)

    # If fields are cell-centered, convert them to point-centered
    g = grid
    needs_convert = ("U" in g.cell_data) or ("p" in g.cell_data)
    if needs_convert:
        # This interpolates cell_data onto points so lengths match g.points
        g = g.cell_data_to_point_data(pass_cell_data=False)

    pts = np.asarray(g.points, dtype=np.float32)

    # Pull strictly from POINT data now
    if "U" not in g.point_data:
        raise KeyError("Field 'U' not found in point_data (after conversion)")

    U = np.asarray(g.point_data["U"], dtype=np.float32)

    if "p" in g.point_data:
        P = np.asarray(g.point_data["p"], dtype=np.float32).reshape(-1, 1)
    else:
        print("[warn] 'p' not found in point_data; filling zeros")
        P = np.zeros((pts.shape[0], 1), dtype=np.float32)

    # Final sanity: make sure lengths match
    n = pts.shape[0]
    if (U.shape[0] != n) or (P.shape[0] != n):
        raise ValueError(f"Length mismatch after conversion: pts={n}, U={U.shape[0]}, P={P.shape[0]}")

    return pts, U, P


# -------------------- Training loop --------------------
def train(volume_path: str):
    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, U, P = load_vtk(volume_path)
    X = X / L_REF
    U = U / U_INF
    P = P / (RHO * U_INF ** 2)

    X_t = torch.from_numpy(X).to(device)
    U_t = torch.from_numpy(U).to(device)
    P_t = torch.from_numpy(P).to(device)

    mins = X_t.min(dim=0).values
    maxs = X_t.max(dim=0).values

    model = MLP(in_dim=3, out_dim=4, width=128, depth=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    nu_eff = torch.tensor(1.0 / RE, dtype=torch.float32, device=device)

    for ep in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        x_coll = torch.rand(BATCH_COLL, 3, device=device) * (maxs - mins) + mins
        r_c, r_m = ns_residuals(model, x_coll, rho=1.0, nu_eff=nu_eff)
        L_pde = (r_c.pow(2).mean()) + (r_m.pow(2).mean())

        if BATCH_DATA > 0:
            idx = torch.randint(0, X_t.shape[0], (BATCH_DATA,), device=device)
            x_d = X_t[idx]
            pred = model(x_d)
            u_pred, p_pred = pred[:, 0:3], pred[:, 3:4]
            L_data = (u_pred - U_t[idx]).pow(2).mean() + (p_pred - P_t[idx]).pow(2).mean()
        else:
            L_data = torch.tensor(0.0, device=device)

        loss = L_pde + L_data
        loss.backward()
        opt.step()

        if ep % 100 == 0 or ep == 1:
            print(f"ep {ep:5d} | L_pde={L_pde.item():.3e} | L_data={L_data.item():.3e} | L={loss.item():.3e}")

    return model

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Steady PINN for motorBike VTK")
    parser.add_argument("--vtk", type=str, required=True, help="Path to volume .vtk/.vtu")
    args = parser.parse_args()
    _ = train(volume_path=args.vtk)