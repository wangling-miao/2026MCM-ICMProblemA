"""
PINN-based Model Correction
============================

Implements Section 4.4.3 of the paper:
- Physics-Informed Neural Network for residual learning
- Hybrid model: Physical model + NN correction
- MC Dropout for uncertainty quantification
"""

import numpy as np

class PINNCorrection:
    """
    Physics-Informed Neural Network for model residual correction.
    
    Implements Eq. 696-702:
    V_pred = V_phys + δV_NN(SOC, I, T)
    
    Uses simple feedforward NN with Tanh activation.
    """
    
    def __init__(self, hidden_sizes=[32, 16], learning_rate=0.001, 
                 w_data=1.0, w_physics=0.5):
        """
        Initialize PINN.
        
        Parameters
        ----------
        hidden_sizes : list
            Sizes of hidden layers
        learning_rate : float
            Learning rate for gradient descent
        w_data : float
            Weight for data loss
        w_physics : float
            Weight for physics loss
        """
        self.hidden_sizes = hidden_sizes
        self.lr = learning_rate
        self.w_data = w_data
        self.w_physics = w_physics
        
        self.weights = []
        self.biases = []
        
        self._initialized = False
    
    def _initialize_weights(self, input_dim=3):
        """Initialize network weights using Xavier initialization."""
        np.random.seed(42)
        
        layer_sizes = [input_dim] + self.hidden_sizes + [1]
        
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros(fan_out)
            
            self.weights.append(W)
            self.biases.append(b)
        
        self._initialized = True
    
    def forward(self, x, dropout_rate=0.0):
        """
        Forward pass through network.
        
        Parameters
        ----------
        x : ndarray
            Input [SOC, I, T] shape (N, 3)
        dropout_rate : float
            Dropout rate for uncertainty estimation
            
        Returns
        -------
        ndarray
            Output correction δV shape (N, 1)
        """
        if not self._initialized:
            self._initialize_weights(x.shape[1])
        
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            
            if i < len(self.weights) - 1:  # Hidden layers
                h = np.tanh(z)  # Tanh activation
                
                # MC Dropout
                if dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - dropout_rate, h.shape)
                    h = h * mask / (1 - dropout_rate)
            else:
                h = z  # Linear output
        
        return h
    
    def compute_loss(self, X, V_meas, V_phys, dV_dSOC_phys):
        """
        Compute combined data + physics loss.
        
        Parameters
        ----------
        X : ndarray
            Input features [SOC, I, T]
        V_meas : ndarray
            Measured voltage
        V_phys : ndarray
            Physics model prediction
        dV_dSOC_phys : ndarray
            Physics model gradient dV/dSOC
            
        Returns
        -------
        float
            Total loss
        """
        # PINN correction
        delta_V = self.forward(X)
        V_pred = V_phys + delta_V.flatten()
        
        # Data loss
        L_data = np.mean((V_pred - V_meas) ** 2)
        
        # Physics loss (gradient consistency)
        # Approximate dδV/dSOC numerically
        eps = 1e-4
        X_plus = X.copy()
        X_plus[:, 0] += eps
        delta_V_plus = self.forward(X_plus)
        
        d_deltaV_dSOC = (delta_V_plus - delta_V) / eps
        
        # Physics constraint: Total gradient should be consistent
        L_physics = np.mean(d_deltaV_dSOC ** 2)  # Penalize large NN gradients
        
        return self.w_data * L_data + self.w_physics * L_physics
    
    def train(self, X, V_meas, V_phys, n_epochs=100, verbose=True):
        """
        Train PINN on data.
        
        Parameters
        ----------
        X : ndarray
            Input features [SOC, I, T] shape (N, 3)
        V_meas : ndarray
            Measured voltages shape (N,)
        V_phys : ndarray
            Physics predictions shape (N,)
        n_epochs : int
            Number of training epochs
        verbose : bool
            Print training progress
        """
        if not self._initialized:
            self._initialize_weights(X.shape[1])
        
        for epoch in range(n_epochs):
            # Forward pass
            delta_V = self.forward(X)
            V_pred = V_phys + delta_V.flatten()
            
            # Compute gradients via finite differences (simplified)
            loss = np.mean((V_pred - V_meas) ** 2)
            
            # Gradient descent update (simplified)
            eps = 1e-5
            for layer_idx in range(len(self.weights)):
                for i in range(self.weights[layer_idx].shape[0]):
                    for j in range(self.weights[layer_idx].shape[1]):
                        # Numerical gradient
                        self.weights[layer_idx][i, j] += eps
                        loss_plus = np.mean((V_phys + self.forward(X).flatten() - V_meas) ** 2)
                        self.weights[layer_idx][i, j] -= 2 * eps
                        loss_minus = np.mean((V_phys + self.forward(X).flatten() - V_meas) ** 2)
                        self.weights[layer_idx][i, j] += eps
                        
                        grad = (loss_plus - loss_minus) / (2 * eps)
                        self.weights[layer_idx][i, j] -= self.lr * grad
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.6f}")
    
    def predict_with_uncertainty(self, X, V_phys, n_samples=50, dropout_rate=0.1):
        """
        Predict with MC Dropout uncertainty.
        
        Parameters
        ----------
        X : ndarray
            Input features
        V_phys : ndarray
            Physics predictions
        n_samples : int
            Number of MC samples
        dropout_rate : float
            Dropout probability
            
        Returns
        -------
        dict
            Mean, std, and prediction intervals
        """
        predictions = []
        
        for _ in range(n_samples):
            delta_V = self.forward(X, dropout_rate=dropout_rate)
            V_pred = V_phys + delta_V.flatten()
            predictions.append(V_pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'lower_95': np.percentile(predictions, 2.5, axis=0),
            'upper_95': np.percentile(predictions, 97.5, axis=0)
        }
    
    def save_weights(self, filepath):
        """Save network weights."""
        np.savez(filepath, 
                 weights=self.weights, 
                 biases=self.biases,
                 hidden_sizes=self.hidden_sizes)
    
    def load_weights(self, filepath):
        """Load network weights."""
        data = np.load(filepath, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        self.hidden_sizes = list(data['hidden_sizes'])
        self._initialized = True
