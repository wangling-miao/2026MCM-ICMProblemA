"""
State Observer Module
=====================

Implements Section 5.1 of the paper (模型求解 - 状态估计算法):
- Extended Kalman Filter (EKF) for SOC estimation (Eq. state_vector, state_equation, measurement_equation)
- Adaptive Extended Kalman Filter (AEKF) with dynamic noise covariance (Eq. adaptive_Q_improved, adaptive_R)

State vector (Eq. state_vector):
    x = [SOC, V1, V2]^T

State transition (Eq. state_transition):
    SOC_{k+1} = SOC_k - (Δt/Q_n) * I_k
    V_{i,k+1} = exp(-Δt/τ_i) * V_{i,k} + (1-exp(-Δt/τ_i)) * R_i * I_k

Measurement function (Eq. measurement_function):
    h = U_ocv(SOC, T) - I*R0(T) - V1 - V2
"""

import numpy as np
from .constants import SimulationParams, R_GAS, T_REF

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for battery state estimation.
    
    Implements the discrete-time state-space model from Section 5.1:
    - State vector: x = [SOC, V1, V2]^T
    - Uses exact discrete solution for RC voltages (避免数值发散)
    """
    
    def __init__(self, battery_model, circuit_model, params=None):
        self.battery = battery_model
        self.circuit = circuit_model
        self.sim = params if params else SimulationParams()
        self.n_states = 3
        self.x_hat = np.array([1.0, 0.0, 0.0])
        self.P = np.diag([1e-2, 1e-4, 1e-4])
        self.Q = np.diag([self.sim.Q_soc, self.sim.Q_V1, self.sim.Q_V2])
        self.R = self.sim.R_meas
        
    def reset(self, SOC_init=1.0):
        self.x_hat = np.array([SOC_init, 0.0, 0.0])
        self.P = np.diag([1e-2, 1e-4, 1e-4])
    
    def state_transition(self, x, I, T, dt):
        """
        State transition function f(x_k, I_k, T_k) from Eq. state_transition.
        
        SOC: SOC_k - (Δt/Q_n) * I_k  (Q_n in Ah, dt in s, I in A)
        V_i: exp(-Δt/τ_i) * V_i + (1-exp(-Δt/τ_i)) * R_i * I  (exact discrete)
        """
        SOC, V1, V2 = x
        # Q_n is in Ah, need to convert to As for time in seconds
        Q_n_As = self.battery.Q_n * 3600.0  # Ah -> As
        tau1, tau2 = self.circuit.tau1(T), self.circuit.tau2(T)
        R1, R2 = self.circuit.R1(T), self.circuit.R2(T)
        
        # SOC update: Eq. state_transition first row
        SOC_next = SOC - (dt / Q_n_As) * I
        exp1, exp2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
        V1_next = V1 * exp1 + R1 * I * (1.0 - exp1)
        V2_next = V2 * exp2 + R2 * I * (1.0 - exp2)
        return np.array([SOC_next, V1_next, V2_next])
    
    def jacobian_A(self, x, I, T, dt):
        tau1, tau2 = self.circuit.tau1(T), self.circuit.tau2(T)
        exp1, exp2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
        return np.array([[1.0, 0.0, 0.0], [0.0, exp1, 0.0], [0.0, 0.0, exp2]])
    
    def measurement_function(self, x, I, T):
        SOC, V1, V2 = x
        OCV = self.circuit.OCV(SOC, T)
        R0 = self.circuit.R0(T, self.battery.delta_SEI)
        return OCV - I * R0 - V1 - V2
    
    def jacobian_C(self, x, I, T):
        dOCV_dSOC = self.circuit.dOCV_dSOC(x[0])
        return np.array([[dOCV_dSOC, -1.0, -1.0]])
    
    def predict(self, I, T, dt):
        self.x_hat = self.state_transition(self.x_hat, I, T, dt)
        A = self.jacobian_A(self.x_hat, I, T, dt)
        self.P = A @ self.P @ A.T + self.Q
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)
    
    def update(self, y_meas, I, T):
        y_pred = self.measurement_function(self.x_hat, I, T)
        innovation = y_meas - y_pred
        C = self.jacobian_C(self.x_hat, I, T)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T / S
        self.x_hat = self.x_hat + K.flatten() * innovation
        I_KC = np.eye(self.n_states) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + (K @ K.T) * self.R
        self.x_hat[0] = np.clip(self.x_hat[0], 0.0, 1.0)
    
    def step(self, y_meas, I, T, dt):
        self.predict(I, T, dt)
        self.update(y_meas, I, T)
        return {'SOC': self.x_hat[0], 'V1': self.x_hat[1], 'V2': self.x_hat[2],
                'SOC_std': np.sqrt(self.P[0, 0])}


class AdaptiveEKF(ExtendedKalmanFilter):
    """Adaptive EKF with dynamic noise covariance (Eq. 676-684)."""
    
    def __init__(self, battery_model, circuit_model, params=None):
        super().__init__(battery_model, circuit_model, params)
        self.lambda_forget = self.sim.lambda_forget
        
    def step(self, y_meas, I, T, dt):
        self.predict(I, T, dt)
        y_pred = self.measurement_function(self.x_hat, I, T)
        innovation = y_meas - y_pred
        
        # Adapt R
        self.R = (1 - self.lambda_forget) * self.R + self.lambda_forget * innovation**2
        self.R = max(self.R, 1e-8)
        
        self.update(y_meas, I, T)
        return {'SOC': self.x_hat[0], 'V1': self.x_hat[1], 'V2': self.x_hat[2],
                'SOC_std': np.sqrt(self.P[0, 0]), 'R': self.R}
