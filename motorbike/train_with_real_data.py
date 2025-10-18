"""
Training Script for Neural Network with Real OpenFOAM motorBike Data
Uses actual simpleFoam simulation results for training
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from real_data_processor import RealMotorBikeDataProcessor
from neural_network_pytorch import MotorBikeNNPyTorch
import os
import torch


def plot_predictions(y_true, y_pred, target_names=['Cd', 'Cl', 'CmPitch'],
                    save_path='real_data_predictions.png'):
    """Plot true vs predicted values for real data"""
    n_targets = len(target_names)
    fig, axes = plt.subplots(1, n_targets, figsize=(6*n_targets, 5))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()],
               [y_true[:, i].min(), y_true[:, i].max()],
               'r--', lw=2.5, label='Perfect Prediction')

        ax.set_xlabel(f'True {name}', fontsize=12)
        ax.set_ylabel(f'Predicted {name}', fontsize=12)
        ax.set_title(f'{name} Prediction (Real Data)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Calculate R-squared
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_true[:, i] - y_pred[:, i]) ** 2))

        # Add metrics text box
        textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.6f}'
        ax.text(0.05, 0.95, textstr,
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
               fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction results saved to {save_path}")
    plt.close()


def plot_time_series_predictions(time, y_true, y_pred, target_names,
                                 save_path='time_series_predictions.png'):
    """Plot predictions vs true values over time"""
    n_targets = len(target_names)
    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 4*n_targets))

    if n_targets == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, target_names)):
        ax.plot(time, y_true[:, i], 'b-', label='True', linewidth=2, alpha=0.7)
        ax.plot(time, y_pred[:, i], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} Time Series Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Time series plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline with real OpenFOAM data"""
    print("=" * 70)
    print("MotorBike CFD Neural Network - Training with Real Data (PyTorch)")
    print("=" * 70)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize data processor
    case_dir = r"C:\Users\evina\AppData\Roaming\Keysight-OpenCFD\OpenFOAM\v2506\msys64\home\ofuser\OpenFOAM\OpenFOAM-v2506\tutorials\incompressible\simpleFoam\motorBike"
    processor = RealMotorBikeDataProcessor(case_dir)

    # Load and parse real data
    print("\n[1] Loading real OpenFOAM simulation data...")
    print("-" * 70)
    coeff_data = processor.parse_coefficient_data()
    solver_data = processor.parse_solver_info()

    # Create ML dataset
    print("\n[2] Creating machine learning dataset...")
    print("-" * 70)
    X, y, feature_names, target_names = processor.create_ml_dataset(
        use_convergence_history=True
    )

    # Get merged dataframe for time information
    merged_df = processor.data['ml_dataset']['merged_df']
    time_values = merged_df['Time'].values

    # Normalize data using StandardScaler
    print("\n[3] Normalizing data...")
    print("-" * 70)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)

    # Split data (preserve time order for validation)
    print("\n[4] Splitting data into train/validation/test sets...")
    print("-" * 70)

    # Use 70% for training, 15% for validation, 15% for testing
    # Keep time order - use later timesteps for validation/testing
    n_samples = len(X_norm)
    train_idx = int(0.70 * n_samples)
    val_idx = int(0.85 * n_samples)

    X_train = X_norm[:train_idx]
    y_train = y_norm[:train_idx]
    X_val = X_norm[train_idx:val_idx]
    y_val = y_norm[train_idx:val_idx]
    X_test = X_norm[val_idx:]
    y_test = y_norm[val_idx:]
    time_test = time_values[val_idx:]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total features: {X_train.shape[1]}")

    # Build and compile model
    print("\n[5] Building neural network...")
    print("-" * 70)
    nn = MotorBikeNNPyTorch(input_dim=X_train.shape[1], output_dim=y_train.shape[1])

    # Use residual architecture for better performance
    nn.build_model(architecture='residual', dropout_rate=0.3)
    print("\nModel Architecture:")
    nn.get_model_summary()

    print("\n[6] Compiling model...")
    nn.compile_model(learning_rate=0.001, optimizer_type='adam', weight_decay=0.0005)

    # Train model
    print("\n[7] Training neural network...")
    print("-" * 70)
    nn.train(
        X_train, y_train,
        X_val, y_val,
        epochs=300,
        batch_size=16,  # Smaller batch size for small dataset
        patience=30,
        verbose=1
    )

    # Evaluate on test set
    print("\n[8] Evaluating model on test set...")
    print("-" * 70)
    test_loss, test_mae = nn.evaluate(X_test, y_test)
    print(f"\nTest Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")

    # Make predictions
    print("\n[9] Generating predictions...")
    y_pred_norm = nn.predict(X_test)

    # Denormalize predictions
    y_pred = scaler_y.inverse_transform(y_pred_norm)
    y_test_denorm = scaler_y.inverse_transform(y_test)

    # Calculate detailed metrics
    print("\n[10] Performance Metrics on Real Data:")
    print("-" * 70)
    for i, name in enumerate(target_names):
        true_vals = y_test_denorm[:, i]
        pred_vals = y_pred[:, i]

        mae = np.mean(np.abs(true_vals - pred_vals))
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        mape = np.mean(np.abs((true_vals - pred_vals) / (np.abs(true_vals) + 1e-8))) * 100

        # R-squared
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Max error
        max_error = np.max(np.abs(true_vals - pred_vals))

        print(f"\n{name}:")
        print(f"  Mean Absolute Error:    {mae:.6f}")
        print(f"  Root Mean Squared Error: {rmse:.6f}")
        print(f"  Mean Abs Percentage Err: {mape:.2f}%")
        print(f"  R² Score:               {r2:.6f}")
        print(f"  Max Error:              {max_error:.6f}")
        print(f"  True range:             [{true_vals.min():.6f}, {true_vals.max():.6f}]")
        print(f"  Predicted range:        [{pred_vals.min():.6f}, {pred_vals.max():.6f}]")

    # Generate plots
    print("\n[11] Generating plots...")
    print("-" * 70)

    # Plot training history
    nn.plot_training_history('real_data_training_history.png')

    # Plot prediction scatter plots
    plot_predictions(y_test_denorm, y_pred, target_names, 'real_data_predictions.png')

    # Plot time series
    plot_time_series_predictions(time_test, y_test_denorm, y_pred, target_names,
                                 'real_data_time_series.png')

    # Plot coefficient and residual histories
    processor.plot_coefficient_history('coefficient_evolution.png')
    processor.plot_residuals('residuals_evolution.png')

    # Save model and scalers
    print("\n[12] Saving model and preprocessing objects...")
    print("-" * 70)
    nn.save_model('motorbike_real_data_model.pth')

    # Save scalers
    import pickle
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)

    # Save feature and target names
    np.savez('model_metadata.npz',
            feature_names=feature_names,
            target_names=target_names)

    print("Scalers saved to scaler_X.pkl and scaler_y.pkl")
    print("Metadata saved to model_metadata.npz")

    # Print summary statistics
    processor.get_summary_statistics()

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - motorbike_real_data_model.pth (trained model)")
    print("  - best_motorbike_model_pytorch.pth (best checkpoint)")
    print("  - scaler_X.pkl, scaler_y.pkl (data scalers)")
    print("  - model_metadata.npz (feature/target names)")
    print("  - real_data_training_history.png")
    print("  - real_data_predictions.png")
    print("  - real_data_time_series.png")
    print("  - coefficient_evolution.png")
    print("  - residuals_evolution.png")

    return nn, processor, scaler_X, scaler_y


if __name__ == "__main__":
    # Create output directory
    os.makedirs('outputs_real_data', exist_ok=True)
    os.chdir('outputs_real_data') if os.path.exists('outputs_real_data') else None

    # Check CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Run training
    model, processor, scaler_X, scaler_y = main()

    print("\n" + "=" * 70)
    print("Model trained on real OpenFOAM data is ready for use!")
    print("=" * 70)
