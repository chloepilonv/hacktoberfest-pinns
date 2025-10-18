"""
Enhanced Data Processor for Real OpenFOAM motorBike simpleFoam Data
Extracts and processes actual CFD simulation results for neural network training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class RealMotorBikeDataProcessor:
    """Process real OpenFOAM motorBike simpleFoam simulation data"""

    def __init__(self, case_dir):
        """
        Initialize data processor for real OpenFOAM data

        Args:
            case_dir: Path to OpenFOAM case directory
        """
        self.case_dir = Path(case_dir)
        self.post_processing_dir = self.case_dir / "postProcessing"
        self.data = {}

    def parse_coefficient_data(self):
        """
        Parse force coefficient data from OpenFOAM postProcessing output

        Returns:
            DataFrame with time and all coefficient columns
        """
        coeff_file = self.post_processing_dir / "forceCoeffs1" / "0" / "coefficient.dat"

        if not coeff_file.exists():
            print(f"Warning: Coefficient file not found at {coeff_file}")
            return None

        # Read the file, skipping comment lines
        data = []
        with open(coeff_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 13:  # Ensure we have all columns
                    # Convert to float
                    row = [float(x) for x in parts]
                    data.append(row)

        # Create DataFrame
        columns = ['Time', 'Cd', 'Cd_f', 'Cd_r', 'Cl', 'Cl_f', 'Cl_r',
                  'CmPitch', 'CmRoll', 'CmYaw', 'Cs', 'Cs_f', 'Cs_r']

        df = pd.DataFrame(data, columns=columns)
        self.data['coefficients'] = df

        print(f"Loaded {len(df)} timesteps of force coefficient data")
        print(f"Time range: {df['Time'].min()} to {df['Time'].max()}")

        return df

    def parse_solver_info(self):
        """
        Parse solver residuals and iteration info

        Returns:
            DataFrame with solver information
        """
        solver_file = self.post_processing_dir / "solverInfo1" / "0" / "solverInfo.dat"

        if not solver_file.exists():
            print(f"Warning: Solver info file not found at {solver_file}")
            return None

        # Read solver info
        data = []
        with open(solver_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 16:
                    # Extract numeric values
                    row = []
                    for i, val in enumerate(parts):
                        if i == 1 or i == 12:  # solver names
                            row.append(val)
                        elif i == 11 or i == 16:  # converged flags
                            row.append(val == 'true')
                        else:
                            try:
                                row.append(float(val))
                            except:
                                row.append(val)
                    data.append(row)

        columns = ['Time', 'U_solver', 'Ux_initial', 'Ux_final', 'Ux_iters',
                  'Uy_initial', 'Uy_final', 'Uy_iters',
                  'Uz_initial', 'Uz_final', 'Uz_iters', 'U_converged',
                  'p_solver', 'p_initial', 'p_final', 'p_iters', 'p_converged']

        df = pd.DataFrame(data, columns=columns)
        self.data['solver_info'] = df

        print(f"Loaded {len(df)} timesteps of solver information")

        return df

    def create_ml_dataset(self, use_convergence_history=True):
        """
        Create machine learning dataset from parsed data

        Args:
            use_convergence_history: Include convergence history features

        Returns:
            X (features), y (targets), feature_names, target_names
        """
        # Load data if not already loaded
        if 'coefficients' not in self.data:
            self.parse_coefficient_data()
        if 'solver_info' not in self.data:
            self.parse_solver_info()

        coeff_df = self.data['coefficients']
        solver_df = self.data['solver_info']

        # Merge datasets on Time
        merged = pd.merge(coeff_df, solver_df, on='Time', how='inner')

        # Extract features
        feature_columns = []

        # Time as a feature
        feature_columns.append('Time')

        # Residual features
        feature_columns.extend(['Ux_initial', 'Ux_final', 'Ux_iters',
                               'Uy_initial', 'Uy_final', 'Uy_iters',
                               'Uz_initial', 'Uz_final', 'Uz_iters',
                               'p_initial', 'p_final', 'p_iters'])

        # Add derived features
        merged['total_U_iters'] = merged['Ux_iters'] + merged['Uy_iters'] + merged['Uz_iters']
        merged['avg_U_residual'] = (merged['Ux_final'] + merged['Uy_final'] + merged['Uz_final']) / 3
        merged['max_U_residual'] = merged[['Ux_final', 'Uy_final', 'Uz_final']].max(axis=1)

        feature_columns.extend(['total_U_iters', 'avg_U_residual', 'max_U_residual'])

        if use_convergence_history:
            # Add rolling statistics (convergence trends)
            window = 5
            for col in ['Ux_final', 'Uy_final', 'Uz_final', 'p_final']:
                merged[f'{col}_mean_{window}'] = merged[col].rolling(window, min_periods=1).mean()
                merged[f'{col}_std_{window}'] = merged[col].rolling(window, min_periods=1).std().fillna(0)
                feature_columns.extend([f'{col}_mean_{window}', f'{col}_std_{window}'])

        # Target variables (force coefficients)
        target_columns = ['Cd', 'Cl', 'CmPitch']

        # Extract features and targets
        X = merged[feature_columns].values
        y = merged[target_columns].values

        self.data['ml_dataset'] = {
            'X': X,
            'y': y,
            'feature_names': feature_columns,
            'target_names': target_columns,
            'merged_df': merged
        }

        print(f"\nCreated ML dataset:")
        print(f"  Features: {X.shape[1]} ({len(feature_columns)} features)")
        print(f"  Targets: {y.shape[1]} ({len(target_columns)} targets)")
        print(f"  Samples: {X.shape[0]}")
        print(f"\nFeature names: {feature_columns}")
        print(f"Target names: {target_columns}")

        return X, y, feature_columns, target_columns

    def plot_coefficient_history(self, save_path='coefficient_history.png'):
        """Plot coefficient evolution over time"""
        if 'coefficients' not in self.data:
            self.parse_coefficient_data()

        df = self.data['coefficients']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Cd
        axes[0, 0].plot(df['Time'], df['Cd'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Drag Coefficient (Cd)')
        axes[0, 0].set_title('Drag Coefficient Evolution')
        axes[0, 0].grid(True, alpha=0.3)

        # Cl
        axes[0, 1].plot(df['Time'], df['Cl'], 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Lift Coefficient (Cl)')
        axes[0, 1].set_title('Lift Coefficient Evolution')
        axes[0, 1].grid(True, alpha=0.3)

        # CmPitch
        axes[1, 0].plot(df['Time'], df['CmPitch'], 'g-', linewidth=2)
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Pitch Moment Coefficient')
        axes[1, 0].set_title('Pitch Moment Evolution')
        axes[1, 0].grid(True, alpha=0.3)

        # Multiple coefficients
        axes[1, 1].plot(df['Time'], df['Cd'], 'b-', label='Cd', linewidth=2)
        axes[1, 1].plot(df['Time'], df['Cl'], 'r-', label='Cl', linewidth=2)
        axes[1, 1].plot(df['Time'], df['CmPitch'], 'g-', label='CmPitch', linewidth=2)
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylabel('Coefficient Value')
        axes[1, 1].set_title('All Coefficients')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Coefficient history plot saved to {save_path}")
        plt.close()

    def plot_residuals(self, save_path='residuals_history.png'):
        """Plot solver residuals over time"""
        if 'solver_info' not in self.data:
            self.parse_solver_info()

        df = self.data['solver_info']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Velocity residuals
        ax1.semilogy(df['Time'], df['Ux_final'], 'b-', label='Ux', linewidth=1.5)
        ax1.semilogy(df['Time'], df['Uy_final'], 'r-', label='Uy', linewidth=1.5)
        ax1.semilogy(df['Time'], df['Uz_final'], 'g-', label='Uz', linewidth=1.5)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Final Residual')
        ax1.set_title('Velocity Component Residuals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Pressure residuals
        ax2.semilogy(df['Time'], df['p_final'], 'k-', label='p', linewidth=2)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Final Residual')
        ax2.set_title('Pressure Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residuals plot saved to {save_path}")
        plt.close()

    def get_summary_statistics(self):
        """Print summary statistics of the data"""
        if 'coefficients' in self.data:
            print("\n" + "=" * 70)
            print("Force Coefficients Summary Statistics")
            print("=" * 70)
            print(self.data['coefficients'].describe())

        if 'solver_info' in self.data:
            print("\n" + "=" * 70)
            print("Solver Residuals Summary Statistics")
            print("=" * 70)
            numeric_cols = self.data['solver_info'].select_dtypes(include=[np.number]).columns
            print(self.data['solver_info'][numeric_cols].describe())
