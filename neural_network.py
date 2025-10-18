"""
Neural Network Model for MotorBike CFD Simulation Prediction
Predicts aerodynamic coefficients from flow parameters
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


class MotorBikeNN:
    """Neural Network for predicting motorBike aerodynamic coefficients"""

    def __init__(self, input_dim=4, output_dim=3):
        """
        Initialize neural network

        Args:
            input_dim: Number of input features (velocity, angle, turbulence, time)
            output_dim: Number of outputs (Cd, Cl, Cm)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None
        self.history = None

    def build_model(self, architecture='standard', dropout_rate=0.2):
        """
        Build neural network architecture

        Args:
            architecture: Type of architecture ('standard', 'deep', 'residual')
            dropout_rate: Dropout rate for regularization

        Returns:
            Keras model
        """
        if architecture == 'standard':
            model = self._build_standard_model(dropout_rate)
        elif architecture == 'deep':
            model = self._build_deep_model(dropout_rate)
        elif architecture == 'residual':
            model = self._build_residual_model(dropout_rate)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.model = model
        return model

    def _build_standard_model(self, dropout_rate):
        """Build standard feedforward neural network"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),

            # First hidden layer
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Second hidden layer
            layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Third hidden layer
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            # Fourth hidden layer
            layers.Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),

            # Output layer
            layers.Dense(self.output_dim, activation='linear')
        ], name='MotorBike_Standard_NN')

        return model

    def _build_deep_model(self, dropout_rate):
        """Build deeper neural network with more layers"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(self.output_dim, activation='linear')
        ], name='MotorBike_Deep_NN')

        return model

    def _build_residual_model(self, dropout_rate):
        """Build residual neural network with skip connections"""
        inputs = layers.Input(shape=(self.input_dim,))

        # Initial dense layer
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        # Residual block 1
        residual = x
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])

        # Residual block 2
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        residual = x
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])

        # Output layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.output_dim, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs,
                          name='MotorBike_Residual_NN')
        return model

    def compile_model(self, learning_rate=0.001, optimizer='adam'):
        """
        Compile the model

        Args:
            learning_rate: Learning rate for optimizer
            optimizer: Optimizer type ('adam', 'sgd', 'rmsprop')
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.model.compile(
            optimizer=opt,
            loss='mse',
            metrics=['mae', 'mse']
        )

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=200, batch_size=32, verbose=1):
        """
        Train the neural network

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level

        Returns:
            Training history
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_motorbike_model.keras',
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True,
                verbose=1
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predicted aerodynamic coefficients
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Test loss and metrics
        """
        return self.model.evaluate(X_test, y_test)

    def plot_training_history(self, save_path='training_history.png'):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        if 'val_mae' in self.history.history:
            ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()

    def save_model(self, filepath='motorbike_model.keras'):
        """Save model to file"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='motorbike_model.keras'):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

    def get_model_summary(self):
        """Print model summary"""
        if self.model is not None:
            return self.model.summary()
        else:
            print("Model not built yet")
