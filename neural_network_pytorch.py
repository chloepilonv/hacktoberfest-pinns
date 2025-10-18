"""
PyTorch Neural Network Model for MotorBike CFD Simulation Prediction
Predicts aerodynamic coefficients from flow parameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class StandardNet(nn.Module):
    """Standard feedforward neural network"""

    def __init__(self, input_dim=4, output_dim=3, dropout_rate=0.2):
        super(StandardNet, self).__init__()

        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second hidden layer
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Third hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Fourth hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Output layer
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class DeepNet(nn.Module):
    """Deeper neural network with more layers"""

    def __init__(self, input_dim=4, output_dim=3, dropout_rate=0.2):
        super(DeepNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResidualNet(nn.Module):
    """Residual neural network with skip connections"""

    def __init__(self, input_dim=4, output_dim=3, dropout_rate=0.2):
        super(ResidualNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.res_block1 = ResidualBlock(128, dropout_rate)

        self.transition1 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.res_block2 = ResidualBlock(256, dropout_rate)

        self.output_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.transition1(x)
        x = self.res_block2(x)
        x = self.output_layers(x)
        return x


class MotorBikeNNPyTorch:
    """PyTorch Neural Network wrapper for training and inference"""

    def __init__(self, input_dim=4, output_dim=3, device=None):
        """
        Initialize neural network

        Args:
            input_dim: Number of input features
            output_dim: Number of outputs
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': []
        }

    def build_model(self, architecture='standard', dropout_rate=0.2):
        """
        Build neural network architecture

        Args:
            architecture: Type of architecture ('standard', 'deep', 'residual')
            dropout_rate: Dropout rate for regularization
        """
        if architecture == 'standard':
            self.model = StandardNet(self.input_dim, self.output_dim, dropout_rate)
        elif architecture == 'deep':
            self.model = DeepNet(self.input_dim, self.output_dim, dropout_rate)
        elif architecture == 'residual':
            self.model = ResidualNet(self.input_dim, self.output_dim, dropout_rate)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model = self.model.to(self.device)
        print(f"\nBuilt {architecture} architecture")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def compile_model(self, learning_rate=0.001, optimizer_type='adam', weight_decay=0.001):
        """
        Setup optimizer and scheduler

        Args:
            learning_rate: Initial learning rate
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            weight_decay: L2 regularization weight
        """
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                       lr=learning_rate,
                                       weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(),
                                      lr=learning_rate,
                                      momentum=0.9,
                                      weight_decay=weight_decay)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10,
            min_lr=1e-7
        )

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Calculate MAE
            mae = torch.mean(torch.abs(outputs - y_batch))

            total_loss += loss.item()
            total_mae += mae.item()

        return total_loss / len(train_loader), total_mae / len(train_loader)

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_mae = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                mae = torch.mean(torch.abs(outputs - y_batch))

                total_loss += loss.item()
                total_mae += mae.item()

        return total_loss / len(val_loader), total_mae / len(val_loader)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=200, batch_size=32, patience=20, verbose=1):
        """
        Train the neural network

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            verbose: Verbosity level
        """
        # Convert to PyTorch tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        print("\nStarting training...")
        print("=" * 70)

        for epoch in range(epochs):
            # Train
            train_loss, train_mae = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)

            # Validate
            if val_loader is not None:
                val_loss, val_mae = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)

                # Learning rate scheduling
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model('best_motorbike_model_pytorch.pth')
                else:
                    patience_counter += 1

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.6f} - MAE: {train_mae:.6f} - "
                          f"Val Loss: {val_loss:.6f} - Val MAE: {val_mae:.6f}")

                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Loss: {train_loss:.6f} - MAE: {train_mae:.6f}")

        print("=" * 70)
        print("Training completed!")

        # Load best model
        if val_loader is not None:
            self.load_model('best_motorbike_model_pytorch.pth')

    def predict(self, X):
        """Make predictions"""
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t)

        return predictions.cpu().numpy()

    def evaluate(self, X_test, y_test):
        """Evaluate model on test data"""
        self.model.eval()
        X_t = torch.FloatTensor(X_test).to(self.device)
        y_t = torch.FloatTensor(y_test).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t)
            loss = self.criterion(predictions, y_t)
            mae = torch.mean(torch.abs(predictions - y_t))

        return loss.item(), mae.item()

    def plot_training_history(self, save_path='training_history_pytorch.png'):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        ax1.plot(self.history['train_loss'], label='Training Loss')
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE
        ax2.plot(self.history['train_mae'], label='Training MAE')
        if self.history['val_mae']:
            ax2.plot(self.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()

    def save_model(self, filepath='motorbike_model_pytorch.pth'):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'history': self.history
        }, filepath)

    def load_model(self, filepath='motorbike_model_pytorch.pth'):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Model loaded from {filepath}")

    def get_model_summary(self):
        """Print model summary"""
        print(self.model)
        print(f"\nTotal parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
