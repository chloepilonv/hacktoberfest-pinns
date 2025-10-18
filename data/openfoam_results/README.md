# OpenFOAM MotorBike Simulation Results

## Simulation Overview

This directory contains results from a complete OpenFOAM CFD simulation of airflow around a motorbike geometry.

### Simulation Details

- **Solver**: simpleFoam (steady-state incompressible flow)
- **OpenFOAM Version**: v2506 (Keysight-OpenCFD Windows build)
- **Geometry**: Standard motorBike tutorial geometry
- **Execution Mode**: Serial (single core)
- **Total Iterations**: 500
- **Turbulence Model**: k-omega SST

### Simulation Steps Completed

1. **Surface Feature Extraction** - Extracted geometric features from motorBike.obj.gz
2. **Base Mesh Generation** (blockMesh) - Created structured background mesh
3. **Detailed Mesh** (snappyHexMesh) - Refined mesh around motorbike geometry with boundary layers
4. **Topology Set** (topoSet) - Defined cell zones
5. **Initial Flow Field** (potentialFoam) - Computed initial velocity field
6. **Mesh Quality Check** (checkMesh) - Verified mesh quality
7. **CFD Simulation** (simpleFoam) - Solved incompressible Navier-Stokes equations for 500 iterations

### Results Summary

#### Final Aerodynamic Coefficients (Time = 500)

- **Drag Coefficient (Cd)**: 0.4168
- **Lift Coefficient (Cl)**: 0.0631

These values represent the normalized aerodynamic forces on the motorbike in the simulated wind tunnel.

### Files Included

- `force_coefficients.dat` - Time history of drag, lift, and moment coefficients (500 time steps)
- `simulation.log` - Complete simulation output log from simpleFoam
- `Allrun-serial` - Bash script used to execute the simulation

### Data Format

The force coefficient file contains the following columns:
1. Time step
2. Cd (total drag coefficient)
3. Cd(f) (front drag)
4. Cd(r) (rear drag)
5. Cl (lift coefficient)
6. Cl(f) (front lift)
7. Cl(r) (rear lift)
8. CmPitch (pitching moment)
9. CmRoll (rolling moment)
10. CmYaw (yawing moment)
11. Cs (side force coefficient)
12. Cs(f) (front side force)
13. Cs(r) (rear side force)

### Use for PINNs Training

This high-fidelity CFD data can be used to:
- Train Physics-Informed Neural Networks for fluid dynamics
- Validate PINN predictions against ground truth CFD results
- Provide boundary conditions and flow field data
- Study aerodynamic force prediction using machine learning

### Simulation Date

October 18, 2025

### Notes

- Simulation converged successfully to steady-state
- All mesh quality checks passed
- No numerical instabilities encountered
- Results are consistent with expected aerodynamic behavior for bluff body flows


