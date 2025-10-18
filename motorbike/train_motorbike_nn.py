"""
Main training script for MotorBike Neural Network
Trains a neural network to predict aerodynamic coefficients from flow parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from data_processor import MotorBikeDataProcessor
from neural_network import MotorBikeNN
import os


def plot_predictions(y_true, y_pred, output_names=['Cd', 'Cl', 'Cm'],
                    save_path='prediction_results.png'):
    """Plot true vs predicted values"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (ax, name) in enumerate(zip(axes, output_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=20)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
               [y_true[:, i].min(), y_true[:, i].max()],
               'r--', lw=2, label='Perfect Prediction')

        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name} Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Calculate R-squared
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction results saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("MotorBike CFD Neural Network Training")
    print("=" * 70)

    # Initialize data processor
    case_dir = r"C:\Users\evina\OneDrive\Documents\repo\motorbike\OpenFOAM-dev\tutorials\incompressibleFluid\motorBikeSteady"
    processor = MotorBikeDataProcessor(case_dir)

    # Try to load real data
    print("\n[1] Loading data...")
    force_data = processor.parse_force_coeffs()

    if force_data is None or len(force_data) < 10:
        print("Real simulation data not available. Generating synthetic data...")
        X, y = processor.create_synthetic_training_data(n_samples=5000)
        print(f"Generated {len(X)} synthetic samples")
    else:
        print(f"Loaded {len(force_data)} real data points")
        # Extract features from real data
        # For now, use synthetic data as real data doesn't have all features
        X, y = processor.create_synthetic_training_data(n_samples=5000)

    # Normalize data
    print("\n[2] Normalizing data...")
    X_norm, y_norm, X_mean, X_std, y_mean, y_std = processor.normalize_data(X, y)

    # Split data
    print("\n[3] Splitting data into train/validation/test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_norm, y_norm, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Build and compile model
    print("\n[4] Building neural network...")
    nn = MotorBikeNN(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Try different architectures
    architectures = ['standard', 'deep', 'residual']
    print(f"Available architectures: {architectures}")
    selected_arch = 'residual'  # Change this to try different architectures
    print(f"Selected architecture: {selected_arch}")

    nn.build_model(architecture=selected_arch, dropout_rate=0.2)
    print("\nModel Architecture:")
    nn.get_model_summary()

    print("\n[5] Compiling model...")
    nn.compile_model(learning_rate=0.001, optimizer='adam')

    # Train model
    print("\n[6] Training neural network...")
    print("-" * 70)
    history = nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        verbose=1
    )

    # Evaluate on test set
    print("\n[7] Evaluating model on test set...")
    print("-" * 70)
    test_results = nn.evaluate(X_test, y_test)
    print(f"\nTest Loss (MSE): {test_results[0]:.6f}")
    print(f"Test MAE: {test_results[1]:.6f}")

    # Make predictions
    print("\n[8] Generating predictions...")
    y_pred_norm = nn.predict(X_test)

    # Denormalize predictions
    y_pred = y_pred_norm * y_std + y_mean
    y_test_denorm = y_test * y_std + y_mean

    # Calculate metrics
    print("\n[9] Performance Metrics:")
    print("-" * 70)
    output_names = ['Cd', 'Cl', 'Cm']
    for i, name in enumerate(output_names):
        mae = np.mean(np.abs(y_test_denorm[:, i] - y_pred[:, i]))
        rmse = np.sqrt(np.mean((y_test_denorm[:, i] - y_pred[:, i]) ** 2))
        mape = np.mean(np.abs((y_test_denorm[:, i] - y_pred[:, i]) /
                             (y_test_denorm[:, i] + 1e-8))) * 100

        print(f"\n{name}:")
        print(f"  MAE:  {mae:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAPE: {mape:.2f}%")

    # Plot results
    print("\n[10] Generating plots...")
    nn.plot_training_history('training_history.png')
    plot_predictions(y_test_denorm, y_pred, output_names, 'prediction_results.png')

    # Save model
    print("\n[11] Saving model...")
    nn.save_model('motorbike_nn_final.keras')

    # Save normalization parameters
    np.savez('normalization_params.npz',
            X_mean=X_mean, X_std=X_std,
            y_mean=y_mean, y_std=y_std)
    print("Normalization parameters saved to normalization_params.npz")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - motorbike_nn_final.keras (trained model)")
    print("  - best_motorbike_model.keras (best checkpoint)")
    print("  - training_history.png (training curves)")
    print("  - prediction_results.png (prediction scatter plots)")
    print("  - normalization_params.npz (normalization parameters)")

    return nn, processor, test_results


if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    os.chdir('outputs') if os.path.exists('outputs') else None

    # Run training
    trained_model, data_processor, results = main()

    print("\n" + "=" * 70)
    print("You can now use the trained model for predictions!")
    print("=" * 70)
