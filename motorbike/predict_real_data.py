"""
Inference Script for Real Data Model
Load model trained on real OpenFOAM data and make predictions
"""

import numpy as np
import torch
import pickle
from neural_network_pytorch import MotorBikeNNPyTorch


def load_real_data_model(model_path='motorbike_real_data_model.pth',
                          scaler_x_path='scaler_X.pkl',
                          scaler_y_path='scaler_y.pkl',
                          metadata_path='model_metadata.npz'):
    """
    Load trained model and preprocessing objects

    Args:
        model_path: Path to saved model
        scaler_x_path: Path to feature scaler
        scaler_y_path: Path to target scaler
        metadata_path: Path to metadata (feature/target names)

    Returns:
        model, scaler_X, scaler_y, metadata
    """
    # Load metadata
    metadata = np.load(metadata_path, allow_pickle=True)
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']

    input_dim = len(feature_names)
    output_dim = len(target_names)

    # Initialize and load model
    nn = MotorBikeNNPyTorch(input_dim=input_dim, output_dim=output_dim)
    nn.build_model(architecture='residual', dropout_rate=0.3)
    nn.load_model(model_path)

    # Load scalers
    with open(scaler_x_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    print("Model and preprocessing objects loaded successfully!")
    print(f"Input features ({input_dim}): {list(feature_names)}")
    print(f"Output targets ({output_dim}): {list(target_names)}\n")

    return nn, scaler_X, scaler_y, {'feature_names': feature_names, 'target_names': target_names}


def predict_from_residuals(residual_data, model, scaler_X, scaler_y, metadata):
    """
    Make prediction from solver residual data

    Args:
        residual_data: Dictionary with residual information
        model: Trained neural network
        scaler_X: Feature scaler
        scaler_y: Target scaler
        metadata: Model metadata

    Returns:
        Dictionary with predictions
    """
    feature_names = metadata['feature_names']
    target_names = metadata['target_names']

    # Build feature vector based on required features
    # This is a simplified example - adjust based on your actual features
    features = []

    # Map residual_data to feature vector in correct order
    feature_dict = {}
    for key, value in residual_data.items():
        feature_dict[key] = value

    # Build feature vector in the correct order
    for fname in feature_names:
        if fname in feature_dict:
            features.append(feature_dict[fname])
        else:
            # Handle derived features or missing values
            if fname == 'total_U_iters':
                features.append(feature_dict.get('Ux_iters', 0) +
                              feature_dict.get('Uy_iters', 0) +
                              feature_dict.get('Uz_iters', 0))
            elif fname == 'avg_U_residual':
                features.append((feature_dict.get('Ux_final', 0) +
                               feature_dict.get('Uy_final', 0) +
                               feature_dict.get('Uz_final', 0)) / 3)
            elif fname == 'max_U_residual':
                features.append(max(feature_dict.get('Ux_final', 0),
                                  feature_dict.get('Uy_final', 0),
                                  feature_dict.get('Uz_final', 0)))
            else:
                # Default to 0 for missing features (rolling stats)
                features.append(0.0)

    X = np.array([features])

    # Normalize
    X_norm = scaler_X.transform(X)

    # Predict
    y_pred_norm = model.predict(X_norm)

    # Denormalize
    y_pred = scaler_y.inverse_transform(y_pred_norm)

    # Create result dictionary
    results = {}
    for i, name in enumerate(target_names):
        results[name] = float(y_pred[0, i])

    return results


def main():
    """Demo prediction with real data model"""
    print("=" * 70)
    print("Real Data Model Inference")
    print("=" * 70)

    print("\nLoading model trained on real OpenFOAM data...")
    model, scaler_X, scaler_y, metadata = load_real_data_model(
        'outputs_real_data/motorbike_real_data_model.pth',
        'outputs_real_data/scaler_X.pkl',
        'outputs_real_data/scaler_y.pkl',
        'outputs_real_data/model_metadata.npz'
    )

    # Example predictions based on solver state
    print("=" * 70)
    print("Example Predictions")
    print("=" * 70)

    # Example 1: Early iteration (high residuals)
    print("\nExample 1: Early Iteration (High Residuals)")
    print("-" * 70)
    residual_data_1 = {
        'Time': 10,
        'Ux_initial': 0.048, 'Ux_final': 0.0047, 'Ux_iters': 7,
        'Uy_initial': 0.119, 'Uy_final': 0.0092, 'Uy_iters': 7,
        'Uz_initial': 0.125, 'Uz_final': 0.0102, 'Uz_iters': 7,
        'p_initial': 0.0259, 'p_final': 0.000125, 'p_iters': 4
    }

    print("Solver state:")
    for key, val in residual_data_1.items():
        print(f"  {key}: {val}")

    results_1 = predict_from_residuals(residual_data_1, model, scaler_X, scaler_y, metadata)
    print("\nPredicted coefficients:")
    for name, value in results_1.items():
        print(f"  {name}: {value:.6f}")

    # Example 2: Mid iteration (medium residuals)
    print("\n\nExample 2: Mid Iteration (Medium Residuals)")
    print("-" * 70)
    residual_data_2 = {
        'Time': 100,
        'Ux_initial': 0.0019, 'Ux_final': 0.000148, 'Ux_iters': 8,
        'Uy_initial': 0.0387, 'Uy_final': 0.0033, 'Uy_iters': 7,
        'Uz_initial': 0.0404, 'Uz_final': 0.0034, 'Uz_iters': 7,
        'p_initial': 0.0136, 'p_final': 0.0000634, 'p_iters': 4
    }

    print("Solver state:")
    for key, val in residual_data_2.items():
        print(f"  {key}: {val}")

    results_2 = predict_from_residuals(residual_data_2, model, scaler_X, scaler_y, metadata)
    print("\nPredicted coefficients:")
    for name, value in results_2.items():
        print(f"  {name}: {value:.6f}")

    # Example 3: Late iteration (low residuals, converged)
    print("\n\nExample 3: Late Iteration (Low Residuals, Converged)")
    print("-" * 70)
    residual_data_3 = {
        'Time': 400,
        'Ux_initial': 0.000539, 'Ux_final': 0.0000489, 'Ux_iters': 8,
        'Uy_initial': 0.0123, 'Uy_final': 0.00105, 'Uy_iters': 7,
        'Uz_initial': 0.0140, 'Uz_final': 0.00119, 'Uz_iters': 7,
        'p_initial': 0.00615, 'p_final': 0.0000304, 'p_iters': 4
    }

    print("Solver state:")
    for key, val in residual_data_3.items():
        print(f"  {key}: {val}")

    results_3 = predict_from_residuals(residual_data_3, model, scaler_X, scaler_y, metadata)
    print("\nPredicted coefficients:")
    for name, value in results_3.items():
        print(f"  {name}: {value:.6f}")

    print("\n" + "=" * 70)
    print("\nNote: This model predicts aerodynamic coefficients based on")
    print("solver convergence state (residuals and iteration counts).")
    print("It was trained on real OpenFOAM motorBike simpleFoam data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
