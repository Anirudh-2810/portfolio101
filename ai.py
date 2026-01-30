import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Literal, Optional, Tuple, List
import json

class NeuralEngine:
    """Advanced neural network with multiple optimizers, regularization, and advanced features"""
    
    def __init__(
        self, 
        learning_rate: float = 0.01, 
        optimizer: Literal['sgd', 'adam', 'rmsprop', 'adamw'] = 'adam',
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        early_stopping_patience: int = 20
    ):
        self.layers = []
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.early_stopping_patience = early_stopping_patience
        
        self.loss_history = []
        self.val_loss_history = []
        self.accuracy_history = []
        self.val_accuracy_history = []
        self.cache = {}
        self._t = 0
        self.training_mode = True
        
        print(f"🚀 Neural Engine initialized with {optimizer.upper()} optimizer")
    
    def add_layer(
        self, 
        neurons: int, 
        activation: Literal['relu', 'sigmoid', 'tanh', 'linear', 'leaky_relu', 'swish'] = 'relu',
        input_size: Optional[int] = None,
        dropout: Optional[float] = None
    ):
        """Add a dense layer with optional dropout"""
        layer = {
            'type': 'dense',
            'neurons': neurons,
            'activation': activation,
            'input_size': input_size,
            'dropout': dropout or self.dropout_rate,
            'W': None,
            'b': None,
            'dW': None,
            'db': None,
            # Optimizer states
            'vW': None,
            'vb': None,
            'mW': None,
            'mb': None,
            'sW': None,  # RMSprop
            'sb': None
        }
        self.layers.append(layer)
        print(f"  ➕ Added {activation.upper()} layer: {neurons} neurons" + 
              (f" (dropout={dropout or self.dropout_rate})" if (dropout or self.dropout_rate) > 0 else ""))
    
    def _init_weights(self, X: np.ndarray):
        """Initialize weights with He/Xavier initialization"""
        input_size = X.shape[1]
        
        for i, layer in enumerate(self.layers):
            if layer['W'] is None:
                prev_size = input_size if i == 0 else self.layers[i-1]['neurons']
                layer['input_size'] = prev_size
                
                # He initialization for ReLU, Xavier for others
                if 'relu' in layer['activation']:
                    scale = np.sqrt(2.0 / prev_size)
                else:
                    scale = np.sqrt(1.0 / prev_size)
                
                layer['W'] = np.random.randn(layer['neurons'], prev_size) * scale
                layer['b'] = np.zeros((1, layer['neurons']))
                
                # Initialize optimizer states
                layer['mW'] = np.zeros_like(layer['W'])
                layer['mb'] = np.zeros_like(layer['b'])
                layer['vW'] = np.zeros_like(layer['W'])
                layer['vb'] = np.zeros_like(layer['b'])
                layer['sW'] = np.zeros_like(layer['W'])
                layer['sb'] = np.zeros_like(layer['b'])
    
    def _activation(self, z: np.ndarray, act: str) -> np.ndarray:
        """Activation functions"""
        if act == 'relu':
            return np.maximum(0, z)
        elif act == 'leaky_relu':
            return np.where(z > 0, z, 0.01 * z)
        elif act == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif act == 'tanh':
            return np.tanh(z)
        elif act == 'swish':
            return z * (1 / (1 + np.exp(-np.clip(z, -500, 500))))
        return z  # linear
    
    def _activation_deriv(self, z: np.ndarray, act: str) -> np.ndarray:
        """Activation derivatives"""
        if act == 'relu':
            return (z > 0).astype(float)
        elif act == 'leaky_relu':
            return np.where(z > 0, 1, 0.01)
        elif act == 'sigmoid':
            s = self._activation(z, 'sigmoid')
            return s * (1 - s)
        elif act == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif act == 'swish':
            sig = self._activation(z, 'sigmoid')
            return sig + z * sig * (1 - sig)
        return np.ones_like(z)
    
    def _dropout_mask(self, shape: Tuple, rate: float) -> np.ndarray:
        """Generate dropout mask"""
        if not self.training_mode or rate == 0:
            return np.ones(shape)
        mask = (np.random.rand(*shape) > rate) / (1 - rate)
        return mask
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass with dropout support"""
        self.training_mode = training
        self.cache = {'Z': [], 'A': [X], 'dropout_masks': []}
        self._init_weights(X)
        
        for i, layer in enumerate(self.layers):
            Z = np.dot(self.cache['A'][-1], layer['W'].T) + layer['b']
            A = self._activation(Z, layer['activation'])
            
            # Apply dropout (except last layer)
            if i < len(self.layers) - 1 and layer['dropout'] > 0:
                mask = self._dropout_mask(A.shape, layer['dropout'])
                A = A * mask
                self.cache['dropout_masks'].append(mask)
            else:
                self.cache['dropout_masks'].append(np.ones_like(A))
            
            self.cache['Z'].append(Z)
            self.cache['A'].append(A)
        
        return self.cache['A'][-1]
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray):
        """Backpropagation with L2 regularization"""
        m = X.shape[0]
        dA = output - y
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # Apply dropout mask
            if i < len(self.layers) - 1:
                dA = dA * self.cache['dropout_masks'][i]
            
            dZ = dA * self._activation_deriv(self.cache['Z'][i], layer['activation'])
            
            # Gradients with L2 regularization
            dW = (np.dot(dZ.T, self.cache['A'][i]) / m) + (self.l2_lambda * layer['W'])
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            layer['dW'] = dW
            layer['db'] = db
            
            if i > 0:
                dA = np.dot(dZ, layer['W'])
        
        self._update_weights()
    
    def _update_weights(self):
        """Update weights using selected optimizer"""
        if self.optimizer == 'sgd':
            self._sgd_update()
        elif self.optimizer == 'adam':
            self._adam_update()
        elif self.optimizer == 'adamw':
            self._adamw_update()
        elif self.optimizer == 'rmsprop':
            self._rmsprop_update()
    
    def _sgd_update(self):
        """Stochastic Gradient Descent"""
        for layer in self.layers:
            layer['W'] -= self.learning_rate * layer['dW']
            layer['b'] -= self.learning_rate * layer['db']
    
    def _adam_update(self):
        """Adam optimizer"""
        self._t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        
        for layer in self.layers:
            # Momentum
            layer['mW'] = beta1 * layer['mW'] + (1 - beta1) * layer['dW']
            layer['mb'] = beta1 * layer['mb'] + (1 - beta1) * layer['db']
            
            # Velocity
            layer['vW'] = beta2 * layer['vW'] + (1 - beta2) * (layer['dW'] ** 2)
            layer['vb'] = beta2 * layer['vb'] + (1 - beta2) * (layer['db'] ** 2)
            
            # Bias correction
            mW_hat = layer['mW'] / (1 - beta1 ** self._t)
            mb_hat = layer['mb'] / (1 - beta1 ** self._t)
            vW_hat = layer['vW'] / (1 - beta2 ** self._t)
            vb_hat = layer['vb'] / (1 - beta2 ** self._t)
            
            # Update
            layer['W'] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            layer['b'] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
    
    def _adamw_update(self):
        """AdamW optimizer (Adam with decoupled weight decay)"""
        self._t += 1
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        weight_decay = 0.01
        
        for layer in self.layers:
            layer['mW'] = beta1 * layer['mW'] + (1 - beta1) * layer['dW']
            layer['mb'] = beta1 * layer['mb'] + (1 - beta1) * layer['db']
            layer['vW'] = beta2 * layer['vW'] + (1 - beta2) * (layer['dW'] ** 2)
            layer['vb'] = beta2 * layer['vb'] + (1 - beta2) * (layer['db'] ** 2)
            
            mW_hat = layer['mW'] / (1 - beta1 ** self._t)
            mb_hat = layer['mb'] / (1 - beta1 ** self._t)
            vW_hat = layer['vW'] / (1 - beta2 ** self._t)
            vb_hat = layer['vb'] / (1 - beta2 ** self._t)
            
            # AdamW: separate weight decay
            layer['W'] = layer['W'] * (1 - self.learning_rate * weight_decay)
            layer['W'] -= self.learning_rate * mW_hat / (np.sqrt(vW_hat) + eps)
            layer['b'] -= self.learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)
    
    def _rmsprop_update(self):
        """RMSprop optimizer"""
        beta = 0.9
        eps = 1e-8
        
        for layer in self.layers:
            layer['sW'] = beta * layer['sW'] + (1 - beta) * (layer['dW'] ** 2)
            layer['sb'] = beta * layer['sb'] + (1 - beta) * (layer['db'] ** 2)
            
            layer['W'] -= self.learning_rate * layer['dW'] / (np.sqrt(layer['sW']) + eps)
            layer['b'] -= self.learning_rate * layer['db'] / (np.sqrt(layer['sb']) + eps)
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy for binary/multiclass classification"""
        if y_true.shape[1] == 1:  # Binary
            predictions = (y_pred > 0.5).astype(float)
        else:  # Multiclass
            predictions = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 100, 
        batch_size: int = 32, 
        val_split: float = 0.2,
        verbose: bool = True,
        plot: bool = True
    ):
        """Train the neural engine with early stopping"""
        n = X.shape[0]
        val_size = int(n * val_split)
        
        # Split data
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]
        X_train = X_shuffled[:-val_size] if val_size > 0 else X_shuffled
        y_train = y_shuffled[:-val_size] if val_size > 0 else y_shuffled
        X_val = X_shuffled[-val_size:] if val_size > 0 else None
        y_val = y_shuffled[-val_size:] if val_size > 0 else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🏋️  Training on {len(X_train)} samples, validating on {len(X_val) if X_val is not None else 0}")
        
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train))
            epoch_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i+batch_size]
                Xb, yb = X_train[batch_idx], y_train[batch_idx]
                
                output = self.forward(Xb, training=True)
                self.backward(Xb, yb, output)
                epoch_loss += np.mean((output - yb) ** 2)
            
            # Training metrics
            train_output = self.forward(X_train, training=False)
            train_loss = np.mean((train_output - y_train) ** 2)
            train_acc = self._compute_accuracy(y_train, train_output)
            
            self.loss_history.append(train_loss)
            self.accuracy_history.append(train_acc)
            
            # Validation metrics
            if X_val is not None:
                val_output = self.forward(X_val, training=False)
                val_loss = np.mean((val_output - y_val) ** 2)
                val_acc = self._compute_accuracy(y_val, val_output)
                
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_acc)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch:4d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        
        print(f"✅ Training complete!")
        
        if plot:
            self.plot_training_history()
    
    def plot_training_history(self):
        """Visualize training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.loss_history, label='Train Loss', linewidth=2)
        if self.val_loss_history:
            ax1.plot(self.val_loss_history, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.accuracy_history, label='Train Accuracy', linewidth=2)
        if self.val_accuracy_history:
            ax2.plot(self.val_accuracy_history, label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training History - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.forward(X, training=False)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        loss = np.mean((predictions - y) ** 2)
        accuracy = self._compute_accuracy(y, predictions)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def save(self, path: str):
        """Save model weights and architecture"""
        state = {
            'layers': [{
                'neurons': l['neurons'],
                'activation': l['activation'],
                'dropout': l['dropout'],
                'W': l['W'],
                'b': l['b']
            } for l in self.layers],
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'l2_lambda': self.l2_lambda
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights and architecture"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.optimizer = state['optimizer']
        self.learning_rate = state['learning_rate']
        self.l2_lambda = state['l2_lambda']
        
        self.layers = []
        for layer_state in state['layers']:
            self.add_layer(
                layer_state['neurons'],
                layer_state['activation'],
                dropout=layer_state['dropout']
            )
            # Manually set weights
            self.layers[-1]['W'] = layer_state['W']
            self.layers[-1]['b'] = layer_state['b']
        
        print(f"📂 Model loaded from {path}")
    
    def summary(self):
        """Print model architecture"""
        print("\n" + "="*60)
        print("NEURAL ENGINE ARCHITECTURE")
        print("="*60)
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            if layer['W'] is not None:
                params = layer['W'].size + layer['b'].size
                total_params += params
                print(f"Layer {i+1}: {layer['activation'].upper():10s} | "
                      f"Neurons: {layer['neurons']:4d} | "
                      f"Params: {params:,}")
        
        print("="*60)
        print(f"Total Parameters: {total_params:,}")
        print(f"Optimizer: {self.optimizer.upper()}")
        print(f"Learning Rate: {self.learning_rate}")
        if self.l2_lambda > 0:
            print(f"L2 Regularization: {self.l2_lambda}")
        if self.dropout_rate > 0:
            print(f"Dropout Rate: {self.dropout_rate}")
        print("="*60 + "\n")


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n🎯 Example 1: XOR Problem (Classic Neural Network Test)")
    print("-" * 60)
    
    # Create engine
    engine = NeuralEngine(
        learning_rate=0.1,
        optimizer='adam',
        l2_lambda=0.001,
        dropout_rate=0.0
    )
    
    # Build architecture
    engine.add_layer(16, 'relu')
    engine.add_layer(8, 'relu')
    engine.add_layer(1, 'sigmoid')
    
    engine.summary()
    
    # XOR dataset
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Train
    engine.fit(X, y, epochs=500, batch_size=2, val_split=0.0, verbose=True)
    
    # Test
    results = engine.evaluate(X, y)
    print(f"\n📊 Final Results:")
    print(f"   Loss: {results['loss']:.4f}")
    print(f"   Accuracy: {results['accuracy']:.4f}")
    print(f"\n   Predictions:")
    for i, (inp, pred, target) in enumerate(zip(X, results['predictions'], y)):
        print(f"   {inp} → {pred[0]:.3f} (target: {target[0]})")
    
    # Save model
    engine.save('xor_model.pkl')
    
    print("\n✅ Neural Engine Ready for Production!")
