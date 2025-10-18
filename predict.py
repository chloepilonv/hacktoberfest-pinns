"""
Inference script for MotorBike Neural Network
Load trained model and make predictions on new data
"""

import numpy as np
from neural_network import MotorBikeNN


def load_model_and_params(model_path='motorbike_nn_final.keras',
                          params_path='normalization_params.npz'):
    """
    Load trained model and normalization parameters

    Args:
        model_path: Path to saved model
        params_path: Path to normalization parameters

    Returns:
        model, normalization_params
    """
    # Load model
    nn = MotorBikeNN()
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


def main():
    """Demo prediction"""
    print("Loading trained model...")
    model, params = load_model_and_params(
        'outputs/motorbike_nn_final.keras',
        'outputs/normalization_params.npz'
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
    print("Aerodynamic Coefficient Predictions")
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

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
