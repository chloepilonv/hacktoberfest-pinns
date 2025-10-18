"""
PyTorch Inference script for MotorBike Neural Network
Load trained model and make predictions on new data
"""

import numpy as np
import torch
from neural_network_pytorch import MotorBikeNNPyTorch, StandardNet, DeepNet, ResidualNet


def load_model_and_params(model_path='motorbike_nn_final_pytorch.pth',
                          params_path='normalization_params_pytorch.npz',
                          architecture='residual',
                          input_dim=4,
                          output_dim=3):
    """
    Load trained model and normalization parameters

    Args:
        model_path: Path to saved model
        params_path: Path to normalization parameters
        architecture: Model architecture used
        input_dim: Input dimension
        output_dim: Output dimension

    Returns:
        model, normalization_params
    """
    # Initialize model wrapper
    nn = MotorBikeNNPyTorch(input_dim=input_dim, output_dim=output_dim)

    # Build architecture
    nn.build_model(architecture=architecture, dropout_rate=0.2)

    # Load trained weights
    nn.load_model(model_path)

    # Load normalization parameters
    params = np.load(params_path)

    return nn, params


def predict_aerodynamic_coeffs(velocity, angle_of_attack, turbulence_intensity,
                               time_step, model, norm_params):
    """
    Predict aerodynamic coefficients for given flow conditions

    Args:
        velocity: Flow velocity (m/s)
        angle_of_attack: Angle of attack (degrees)
        turbulence_intensity: Turbulence intensity (0-1)
        time_step: Time step
        model: Trained neural network
        norm_params: Normalization parameters

    Returns:
        Dictionary with Cd, Cl, Cm predictions
    """
    # Create input array
    X = np.array([[velocity, angle_of_attack, turbulence_intensity, time_step]])

    # Normalize
    X_norm = (X - norm_params['X_mean']) / (norm_params['X_std'] + 1e-8)

    # Predict
    y_pred_norm = model.predict(X_norm)

    # Denormalize
    y_pred = y_pred_norm * norm_params['y_std'] + norm_params['y_mean']

    return {
        'Cd': float(y_pred[0, 0]),
        'Cl': float(y_pred[0, 1]),
        'Cm': float(y_pred[0, 2])
    }


def batch_predict(velocities, angles, turbulences, time_steps, model, norm_params):
    """
    Batch prediction for multiple flow conditions

    Args:
        velocities: Array of flow velocities
        angles: Array of angles of attack
        turbulences: Array of turbulence intensities
        time_steps: Array of time steps
        model: Trained neural network
        norm_params: Normalization parameters

    Returns:
        Dictionary with arrays of Cd, Cl, Cm predictions
    """
    # Create input array
    X = np.column_stack([velocities, angles, turbulences, time_steps])

    # Normalize
    X_norm = (X - norm_params['X_mean']) / (norm_params['X_std'] + 1e-8)

    # Predict
    y_pred_norm = model.predict(X_norm)

    # Denormalize
    y_pred = y_pred_norm * norm_params['y_std'] + norm_params['y_mean']

    return {
        'Cd': y_pred[:, 0],
        'Cl': y_pred[:, 1],
        'Cm': y_pred[:, 2]
    }


def main():
    """Demo prediction"""
    print("=" * 70)
    print("PyTorch Model Inference")
    print("=" * 70)

    print("\nLoading trained model...")
    model, params = load_model_and_params(
        'outputs/motorbike_nn_final_pytorch.pth',
        'outputs/normalization_params_pytorch.npz',
        architecture='residual'  # Must match the trained model
    )
    print("Model loaded successfully!\n")

    # Example predictions
    test_cases = [
        {'velocity': 20.0, 'angle': 0.0, 'turbulence': 0.05, 'time': 100},
        {'velocity': 25.0, 'angle': 2.5, 'turbulence': 0.03, 'time': 200},
        {'velocity': 18.0, 'angle': -1.0, 'turbulence': 0.08, 'time': 300},
        {'velocity': 22.0, 'angle': 5.0, 'turbulence': 0.04, 'time': 400},
    ]

    print("=" * 70)
    print("Single Predictions")
    print("=" * 70)

    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Velocity: {case['velocity']} m/s")
        print(f"  Angle of Attack: {case['angle']}°")
        print(f"  Turbulence Intensity: {case['turbulence']}")
        print(f"  Time Step: {case['time']}")

        results = predict_aerodynamic_coeffs(
            case['velocity'],
            case['angle'],
            case['turbulence'],
            case['time'],
            model,
            params
        )

        print(f"\n  Predictions:")
        print(f"    Drag Coefficient (Cd):   {results['Cd']:.6f}")
        print(f"    Lift Coefficient (Cl):   {results['Cl']:.6f}")
        print(f"    Moment Coefficient (Cm): {results['Cm']:.6f}")
        print("-" * 70)

    # Batch prediction example
    print("\n" + "=" * 70)
    print("Batch Prediction Example")
    print("=" * 70)

    velocities = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
    angles = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    turbulences = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    time_steps = np.array([100, 150, 200, 250, 300])

    batch_results = batch_predict(velocities, angles, turbulences, time_steps, model, params)

    print(f"\nPredicting for {len(velocities)} cases...")
    print(f"\nResults:")
    print(f"{'Velocity':>10} {'Angle':>8} {'Cd':>10} {'Cl':>10} {'Cm':>10}")
    print("-" * 70)
    for i in range(len(velocities)):
        print(f"{velocities[i]:>10.1f} {angles[i]:>8.1f} "
              f"{batch_results['Cd'][i]:>10.6f} "
              f"{batch_results['Cl'][i]:>10.6f} "
              f"{batch_results['Cm'][i]:>10.6f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
