# src/predict.py
import argparse, os
import numpy as np
import torch, torch.nn as nn
import pyvista as pv

# must match training
U_INF, L_REF, RHO = 30.0, 1.0, 1.225

class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=4, width=128, depth=6, act=nn.SiLU):
        super().__init__()
        layers=[nn.Linear(in_dim,width),act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width,width),act()]
        layers += [nn.Linear(width,out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def main(vtk_in, weights, out_csv=None, out_vtk=None):
    # load query points (here: reuse the CFD mesh points)
    grid = pv.read(vtk_in)
    pts = np.asarray(grid.points, dtype=np.float32)
    x_nd = torch.from_numpy(pts / L_REF).float()

    # load model
    model = MLP()
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(x_nd).numpy()
    U_pred = pred[:, :3] * U_INF
    p_pred = pred[:, 3] * (RHO * U_INF**2)

    # optional: write CSV
    if out_csv:
        import pandas as pd
        df = pd.DataFrame({
            "x": pts[:,0], "y": pts[:,1], "z": pts[:,2],
            "Ux_pred": U_pred[:,0], "Uy_pred": U_pred[:,1], "Uz_pred": U_pred[:,2],
            "p_pred": p_pred
        })
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")

    # optional: write VTK PolyData for ParaView
    if out_vtk:
        pdset = pv.PolyData(pts)
        pdset["U_pred"] = U_pred
        pdset["p_pred"] = p_pred
        os.makedirs(os.path.dirname(out_vtk), exist_ok=True)
        pdset.save(out_vtk)
        print(f"Wrote {out_vtk}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vtk", required=True, help="Input VTK to reuse its points")
    ap.add_argument("--weights", default="../models/motorbike_pinn.pt")
    ap.add_argument("--out-csv", default="../data/predictions/pred_points.csv")
    ap.add_argument("--out-vtk", default="../data/predictions/pred_points.vtp")
    args = ap.parse_args()
    main(args.vtk, args.weights, args.out_csv, args.out_vtk)
