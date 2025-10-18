"""
Data processor for OpenFOAM motorBike steady-state simulation
Extracts and preprocesses CFD simulation data for neural network training
"""

import numpy as np
import pandas as pd
import os
import re
from pathlib import Path


class MotorBikeDataProcessor:
    """Process OpenFOAM motorBike simulation data for ML applications"""

    def __init__(self, case_dir):
        """
        Initialize data processor

        Args:
            case_dir: Path to OpenFOAM case directory
        """
        self.case_dir = Path(case_dir)
        self.post_processing_dir = self.case_dir / "postProcessing"
        self.data = {}

    def parse_force_coeffs(self, force_coeffs_file=None):
        """
        Parse forceCoeffs data from OpenFOAM output

        Args:
            force_coeffs_file: Path to forceCoeffs.dat file

        Returns:
            DataFrame with time, Cd, Cl, Cm columns
        """
        if force_coeffs_file is None:
            # Search for forceCoeffs file
            force_dir = self.post_processing_dir / "forceCoeffs"
            if force_dir.exists():
                files = list(force_dir.glob("**/coefficient.dat"))
                if files:
                    force_coeffs_file = files[0]

        if force_coeffs_file is None or not Path(force_coeffs_file).exists():
            print(f"Warning: Force coefficients file not found")
            return None

        # Read force coefficients
        data = []
        with open(force_coeffs_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # Time, Cd, Cl, Cm (approximately)
                    data.append([float(parts[0]), float(parts[1]),
                                float(parts[2]), float(parts[3])])

        df = pd.DataFrame(data, columns=['time', 'Cd', 'Cl', 'Cm'])
        self.data['force_coeffs'] = df
        return df

    def parse_log_residuals(self, log_file=None):
        """
        Parse residuals from OpenFOAM log file

        Args:
            log_file: Path to log file (e.g., log.foamRun)

        Returns:
            DataFrame with iteration, Ux, Uy, Uz, p residuals
        """
        if log_file is None:
            log_file = self.case_dir / "log.foamRun"

        if not log_file.exists():
            print(f"Warning: Log file {log_file} not found")
            return None

        residuals = []
        with open(log_file, 'r') as f:
            for line in f:
                # Look for residual lines
                if 'Solving for Ux' in line or 'Solving for Uy' in line or \
                   'Solving for Uz' in line or 'Solving for p' in line:
                    # Extract residual values using regex
                    match = re.search(r'Initial residual = ([\d.e+-]+)', line)
                    if match:
                        residuals.append(float(match.group(1)))

        return np.array(residuals)

    def create_synthetic_training_data(self, n_samples=1000):
        """
        Create synthetic training data based on typical motorBike flow characteristics

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            X: Input features (flow parameters)
            y: Output targets (force coefficients, residuals)
        """
        # Synthetic flow parameters
        # Features: velocity, angle of attack, turbulence intensity, time
        velocity = np.random.uniform(15, 25, n_samples)  # m/s
        angle_of_attack = np.random.uniform(-5, 5, n_samples)  # degrees
        turbulence_intensity = np.random.uniform(0.01, 0.1, n_samples)
        time_steps = np.linspace(0, 500, n_samples)

        # Simulate force coefficients with physics-based relationships
        # Cd increases with angle and turbulence
        Cd_base = 0.3 + 0.02 * np.abs(angle_of_attack) + \
                  0.5 * turbulence_intensity + \
                  0.001 * np.sin(2 * np.pi * time_steps / 500)

        # Cl varies with angle of attack
        Cl_base = 0.1 * np.sin(np.deg2rad(angle_of_attack)) + \
                  0.02 * np.random.randn(n_samples)

        # Cm (moment coefficient)
        Cm_base = -0.05 * angle_of_attack / 10 + \
                  0.01 * np.random.randn(n_samples)

        # Add velocity dependency (Re number effect)
        Re_factor = (velocity / 20) ** 0.2

        Cd = Cd_base * Re_factor + np.random.randn(n_samples) * 0.02
        Cl = Cl_base * Re_factor + np.random.randn(n_samples) * 0.01
        Cm = Cm_base * Re_factor + np.random.randn(n_samples) * 0.005

        # Stack features
        X = np.column_stack([velocity, angle_of_attack,
                            turbulence_intensity, time_steps])
        y = np.column_stack([Cd, Cl, Cm])

        self.data['synthetic_X'] = X
        self.data['synthetic_y'] = y

        return X, y

    def normalize_data(self, X, y):
        """
        Normalize features and targets

        Args:
            X: Input features
            y: Output targets

        Returns:
            X_norm, y_norm, X_mean, X_std, y_mean, y_std
        """
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_norm = (X - X_mean) / (X_std + 1e-8)

        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        y_norm = (y - y_mean) / (y_std + 1e-8)

        self.normalization_params = {
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }

        return X_norm, y_norm, X_mean, X_std, y_mean, y_std

    def save_processed_data(self, output_dir='processed_data'):
        """Save processed data to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for key, value in self.data.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(output_path / f"{key}.csv", index=False)
            elif isinstance(value, np.ndarray):
                np.save(output_path / f"{key}.npy", value)

        print(f"Saved processed data to {output_path}")
