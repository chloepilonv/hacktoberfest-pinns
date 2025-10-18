# MotorBike CFD Neural Network

Neural network for predicting aerodynamic coefficients from OpenFOAM motorBike steady-state simulation data.

## Quick Start Summary

**Latest Training Run (PyTorch):**
- **Framework**: PyTorch 2.9.0+cpu
- **Architecture**: Residual Network (242,179 parameters)
- **Training Data**: 5,000 synthetic samples
- **Training Duration**: 95 epochs (early stopping)
- **Performance Metrics**:
  - Test Loss (MSE): 0.528
  - Test MAE: 0.556
  - Cd MAPE: 4.19%
  - Cl MAPE: 153.47%
  - Cm MAPE: 238.20%

**Generated Files**:
- ✅ `outputs/motorbike_nn_final_pytorch.pth` - Trained model
- ✅ `outputs/normalization_params_pytorch.npz` - Normalization parameters
- ✅ `outputs/training_history_pytorch.png` - Training curves
- ✅ `outputs/prediction_results_pytorch.png` - Prediction scatter plots

**Sample Predictions**:
```
Velocity: 20.0 m/s, Angle: 0.0°, Turbulence: 0.05
  → Cd: 0.3326, Cl: 0.0000, Cm: -0.0003

Velocity: 25.0 m/s, Angle: 2.5°, Turbulence: 0.03
  → Cd: 0.3845, Cl: 0.0057, Cm: -0.0150
```

**System Requirements**:
- Python 3.9-3.14 (TensorFlow requires 3.9-3.12)
- For Python 3.14: Use PyTorch implementation only

---

## Overview

This project implements a deep learning model to predict aerodynamic force coefficients (Cd, Cl, Cm) for a motorbike based on flow parameters:

- **Input Features**: velocity, angle of attack, turbulence intensity, time step
- **Output Predictions**: drag coefficient (Cd), lift coefficient (Cl), moment coefficient (Cm)
- **Frameworks**: Both TensorFlow/Keras and PyTorch implementations available

## Project Structure

```
motorbike/
├── data_processor.py                    # Synthetic data generation
├── real_data_processor.py               # Real OpenFOAM data extraction
├── neural_network.py                    # TensorFlow/Keras neural network
├── neural_network_pytorch.py            # PyTorch neural network
├── train_motorbike_nn.py                # TensorFlow training (synthetic)
├── train_motorbike_nn_pytorch.py        # PyTorch training (synthetic)
├── train_with_real_data.py              # Training with real OpenFOAM data
├── predict.py                           # TensorFlow inference
├── predict_pytorch.py                   # PyTorch inference
├── predict_real_data.py                 # Inference for real data model
├── requirements.txt                     # TensorFlow dependencies
├── requirements_pytorch.txt             # PyTorch dependencies
├── outputs/                             # Synthetic data training outputs
└── outputs_real_data/                   # Real data training outputs
    ├── motorbike_real_data_model.pth
    ├── scaler_X.pkl, scaler_y.pkl
    ├── real_data_predictions.png
    └── coefficient_evolution.png
```

## Installation

### TensorFlow Version

```bash
pip install -r requirements.txt
```

### PyTorch Version

```bash
pip install -r requirements_pytorch.txt
```

## Usage

### Training the Model

**TensorFlow Version:**
```bash
python train_motorbike_nn.py
```

**PyTorch Version:**
```bash
python train_motorbike_nn_pytorch.py
```

This will:
- Load or generate training data
- Build and train the neural network
- Evaluate performance on test set
- Save the trained model and plots

### Making Predictions

**TensorFlow Version:**
```bash
python predict.py
```

**PyTorch Version:**
```bash
python predict_pytorch.py
```

Or use it in your own code:

**TensorFlow Example:**
```python
from predict import load_model_and_params, predict_aerodynamic_coeffs

# Load model
model, params = load_model_and_params(
    'outputs/motorbike_nn_final.keras',
    'outputs/normalization_params.npz'
)

# Make prediction
results = predict_aerodynamic_coeffs(
    velocity=20.0,              # m/s
    angle_of_attack=0.0,        # degrees
    turbulence_intensity=0.05,  # dimensionless
    time_step=100,
    model=model,
    norm_params=params
)

print(f"Cd = {results['Cd']:.4f}")
print(f"Cl = {results['Cl']:.4f}")
print(f"Cm = {results['Cm']:.4f}")
```

**PyTorch Example:**
```python
from predict_pytorch import load_model_and_params, predict_aerodynamic_coeffs

# Load model
model, params = load_model_and_params(
    'outputs/motorbike_nn_final_pytorch.pth',
    'outputs/normalization_params_pytorch.npz',
    architecture='residual'
)

# Make prediction
results = predict_aerodynamic_coeffs(
    velocity=20.0,
    angle_of_attack=0.0,
    turbulence_intensity=0.05,
    time_step=100,
    model=model,
    norm_params=params
)

print(f"Cd = {results['Cd']:.4f}")
print(f"Cl = {results['Cl']:.4f}")
print(f"Cm = {results['Cm']:.4f}")
```

## Neural Network Architectures

Three architectures are available:

1. **Standard**: 4-layer feedforward network with batch normalization and dropout
2. **Deep**: 6-layer deep network for complex patterns
3. **Residual**: Network with skip connections for better gradient flow

Select architecture in `train_motorbike_nn.py`:

```python
selected_arch = 'residual'  # Options: 'standard', 'deep', 'residual'
```

## Model Performance

The model is trained with:
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with learning rate scheduling
- Regularization: L2 weight decay, dropout, batch normalization
- Early stopping to prevent overfitting

Performance metrics include:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² score for each coefficient

## Data Sources

The project supports two types of data:

### 1. Real OpenFOAM Data (Recommended)

Train on actual simpleFoam simulation results from OpenFOAM motorBike case:

```bash
python train_with_real_data.py
```

This uses real solver residuals and force coefficients from:
- `postProcessing/forceCoeffs1/0/coefficient.dat`
- `postProcessing/solverInfo1/0/solverInfo.dat`

**Features**: Time, solver residuals (Ux, Uy, Uz, p), iteration counts, convergence trends
**Targets**: Cd, Cl, CmPitch

The real data model learns to predict aerodynamic coefficients based on solver convergence state.

### 2. Synthetic Data

For demonstration when real data is unavailable, use physics-based synthetic data:

```bash
python train_motorbike_nn_pytorch.py  # or train_motorbike_nn.py
```

To use real simulation data, ensure the OpenFOAM case has been run and postProcessing data is available.

## OpenFOAM Integration

The data processor can extract data from:
- Force coefficients: `postProcessing/forceCoeffs/`
- Log files: Residuals and solver information
- Time directories: Field data (velocity, pressure)

Point to your OpenFOAM case:

```python
case_dir = "path/to/motorBikeSteady"
processor = MotorBikeDataProcessor(case_dir)
force_data = processor.parse_force_coeffs()
```

## Outputs

**TensorFlow Training generates:**
- `motorbike_nn_final.keras`: Final trained model
- `best_motorbike_model.keras`: Best checkpoint during training
- `training_history.png`: Loss and MAE curves
- `prediction_results.png`: Scatter plots of predictions vs true values
- `normalization_params.npz`: Normalization statistics for inference

**PyTorch Training generates:**
- `motorbike_nn_final_pytorch.pth`: Final trained model
- `best_motorbike_model_pytorch.pth`: Best checkpoint during training
- `training_history_pytorch.png`: Loss and MAE curves
- `prediction_results_pytorch.png`: Scatter plots of predictions vs true values
- `normalization_params_pytorch.npz`: Normalization statistics for inference

**Real Data Training generates:**
- `motorbike_real_data_model.pth`: Model trained on real OpenFOAM data
- `scaler_X.pkl`, `scaler_y.pkl`: StandardScaler objects for preprocessing
- `model_metadata.npz`: Feature and target names
- `real_data_predictions.png`: Prediction accuracy plots
- `real_data_time_series.png`: Time series comparison
- `coefficient_evolution.png`: Coefficient history from simulation
- `residuals_evolution.png`: Solver residual convergence

## Framework Comparison

| Feature | TensorFlow/Keras | PyTorch |
|---------|-----------------|---------|
| Model format | `.keras` | `.pth` |
| Training script | `train_motorbike_nn.py` | `train_motorbike_nn_pytorch.py` |
| GPU support | Automatic | Automatic with CUDA |
| Ease of use | High-level API | More control |
| Production deployment | TensorFlow Serving | TorchServe |

Both implementations provide identical functionality and performance.

## Customization

Modify hyperparameters in `train_motorbike_nn.py` or `train_motorbike_nn_pytorch.py`:
- Network architecture
- Learning rate
- Batch size
- Number of epochs
- Dropout rate
- Number of synthetic samples

## License

This project is for educational and research purposes.
