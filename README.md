# hacktoberfest-pinns
PINNs on Openfoam for hacktoberfest 2025. 

# 🏍️ Motorbike PINN (Physics-Informed Neural Network)

A physics-informed neural network (PINN) surrogate model trained on the OpenFOAM **motorBikeSteady** case.  
The network learns the steady-state incompressible Navier–Stokes solution and predicts velocity and pressure fields `(u, v, w, p)` directly from spatial coordinates.

---

## Quick Start

Clone the repository and run the prediction script using your trained model:

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/motorbike-pinn.git

# 2. Train your model. Here, I used Nvidia B200 as part of the GPU Mode IRL 2025 San Francisco Hackathon.
# Will create the models/motorbike_pinn.pt for weights and step 3
cd src
python3 src/openfoam_pinn.py --vtk /data/VTK/motorBikeSteady_500.vtk

# 3. Run inference on the motorbike case
python3 predict.py --vtk /data/VTK/motorBikeSteady_500.vtk \
                   --weights /models/motorbike_pinn.pt \
                   --out-csv /data/predictions/pred_points.csv \
                   --out-vtk /data/predictions/pred_points.vtp
```


---

### Arguments

| Flag | Description |
|------|--------------|
| `--vtk` | Path to the input VTK file exported from OpenFOAM (`foamToVTK -latestTime`). |
| `--weights` | Path to the trained PINN model weights (`.pt` file). |
| `--out-csv` | (Optional) CSV export of predictions for analysis. |
| `--out-vtk` | (Optional) VTK/ParaView file containing predicted fields (`U_pred`, `p_pred`). |

---

### Output

After running the command, the results will be stored in:

```
data/predictions/
├── pred_points.csv   # Numerical results (x, y, z, Ux_pred, Uy_pred, Uz_pred, p_pred)
└── pred_points.vtp   # Visualization file for ParaView
```

Open the `.vtp` file in **ParaView** to visualize the predicted velocity and pressure fields.

---

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Repository Structure

```
motorbike-pinn/
├─ data/
│  └─ VTK/                      # OpenFOAM exported data
├─ models/
│  └─ openfoam_pinn.pt         # trained PINN weights
├─ src/
│  ├─ openfoam_pinn.py         # training script
│  └─ predictopenfoam_pinn.py                # inference script
└─ requirements.txt
```

---

### Credits

- **OpenFOAM motorBike** case (steady, `simpleFoam`)  
- **PyTorch**, **PyVista**, **MeshIO**  
- Developed for research on **Physics-Informed Neural Networks (PINNs)** for CFD surrogate modeling.

