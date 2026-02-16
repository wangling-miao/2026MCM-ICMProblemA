"""
Coupled Battery System
======================

Integrates all model components for complete system simulation:
- Battery dynamics + Equivalent circuit + Power consumption
- State estimation with EKF/AEKF
- Time-to-Empty (TTE) prediction with Monte Carlo
- First-passage time computation via SDE (Eq. 582-620)
"""

import numpy as np
from .battery_dynamics import BatteryDynamics
from .equivalent_circuit import EquivalentCircuitModel
from .power_consumption import PowerConsumptionModel
from .state_observer import ExtendedKalmanFilter, AdaptiveEKF
from .hmm_scenario import HMMScenarioPredictor
from .constants import SimulationParams, BatteryParams, SmartphoneParams

class CoupledBatterySystem:
    """
    Complete coupled battery-smartphone system for TTE prediction.
    
    Implements the full model from Section 4.3:
    - Continuous-time coupled ODE system (Eq. 582-586)
    - First-passage time formulation for TTE (Eq. 593)
    - Monte Carlo TTE prediction (Eq. 610-620)
    """
    
    def __init__(self, battery_params=None, phone_params=None, sim_params=None):
        """
        Initialize coupled system.
        
        Parameters
        ----------
        battery_params : BatteryParams, optional
        phone_params : SmartphoneParams, optional
        sim_params : SimulationParams, optional
        """
        self.bat_p = battery_params if battery_params else BatteryParams()
        self.phone_p = phone_params if phone_params else SmartphoneParams()
        self.sim_p = sim_params if sim_params else SimulationParams()
        
        # Initialize sub-models
        self.battery = BatteryDynamics(self.bat_p)
        self.circuit = EquivalentCircuitModel(self.bat_p)
        self.power = PowerConsumptionModel(self.phone_p)
        self.hmm = HMMScenarioPredictor()
        
        # State observer
        self.ekf = AdaptiveEKF(self.battery, self.circuit, self.sim_p)
        
        # Simulation state
        self.t = 0.0
        self.history = {
            'time': [], 'SOC': [], 'V_term': [], 'T': [],
            'P_load': [], 'I': [], 'mode': []
        }
    
    def reset(self, SOC_init=1.0, T_init=298.15):
        """Reset system to initial state."""
        self.battery.reset(SOC_init)
        self.circuit.reset(T_init)
        self.power.reset()
        self.ekf.reset(SOC_init)
        self.t = 0.0
        self.history = {k: [] for k in self.history}
    
    def step(self, dt, mode=None, T_amb=298.15):
        """
        Simulate one time step.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        mode : str, optional
            Usage mode. If None, sampled from HMM.
        T_amb : float
            Ambient temperature [K]
            
        Returns
        -------
        dict
            State after step
        """
        # Get usage mode (from HMM if not specified)
        if mode is None:
            mode = self.hmm.predict_next_state()
        
        # Get power consumption for mode
        P_load = self.power.mode_power(mode)
        
        # Current terminal voltage estimate
        V_term = self.circuit.terminal_voltage(
            self.battery.SOC, 0, self.circuit.T, self.battery.delta_SEI
        )
        
        # Convert power to current
        I = self.power.power_to_current(P_load, V_term)
        
        # Update battery dynamics
        self.battery.update_SOC(I, self.circuit.T, dt)
        self.battery.update_SEI(self.circuit.T, V_term, dt)
        
        # Update circuit model (thermal + polarization)
        circuit_out = self.circuit.step(
            self.battery.SOC, I, dt, T_amb, self.battery.delta_SEI
        )
        V_term = circuit_out['V_term']
        
        # Update EKF with measurement
        ekf_out = self.ekf.step(V_term, I, self.circuit.T, dt)
        
        # Update time
        self.t += dt
        
        # Store history
        self.history['time'].append(self.t)
        self.history['SOC'].append(self.battery.SOC)
        self.history['V_term'].append(V_term)
        self.history['T'].append(self.circuit.T)
        self.history['P_load'].append(P_load)
        self.history['I'].append(I)
        self.history['mode'].append(mode)
        
        return {
            'SOC': self.battery.SOC,
            'SOC_est': ekf_out['SOC'],
            'V_term': V_term,
            'T': self.circuit.T,
            'P_load': P_load,
            'I': I,
            'mode': mode,
            't': self.t
        }
    
    def simulate_until_empty(self, dt=1.0, SOC_cutoff=0.05, max_time=86400, T_amb=298.15):
        """
        Simulate until SOC reaches cutoff.
        
        Parameters
        ----------
        dt : float
            Time step [s]
        SOC_cutoff : float
            SOC threshold for empty
        max_time : float
            Maximum simulation time [s]
        T_amb : float
            Ambient temperature [K]
            
        Returns
        -------
        float
            Time to empty [s]
        """
        self.reset()
        TTE = max_time
        
        while self.t < max_time:
            result = self.step(dt, T_amb=T_amb)
            
            if result['SOC'] <= SOC_cutoff:
                TTE = self.t
                break
        
        return TTE
    
    def predict_TTE_deterministic(self, dt=1.0, SOC_cutoff=0.05, mode='Light'):
        """
        Predict TTE under constant load (deterministic).
        
        Implements Eq. 593:
        TTE = Q_n * (SOC - SOC_cutoff) / I_avg
        
        Parameters
        ----------
        dt : float
            Time step [s]
        SOC_cutoff : float
            SOC threshold
        mode : str
            Fixed usage mode
            
        Returns
        -------
        dict
            TTE prediction and trajectory
        """
        self.reset()
        
        while self.battery.SOC > SOC_cutoff and self.t < 86400:
            self.step(dt, mode=mode)
        
        return {
            'TTE': self.t,
            'history': {k: np.array(v) for k, v in self.history.items()}
        }
    
    def predict_TTE_monte_carlo(self, n_samples=100, dt=10.0, SOC_cutoff=0.05, 
                                max_time=43200):
        """
        Monte Carlo TTE prediction with uncertainty.
        
        Implements Eq. 610-620:
        E[TTE] = (1/N) * Σ TTE^(i)
        Var[TTE] = (1/N) * Σ (TTE^(i) - E[TTE])²
        
        Parameters
        ----------
        n_samples : int
            Number of MC samples
        dt : float
            Time step [s]
        SOC_cutoff : float
            SOC threshold
        max_time : float
            Maximum simulation time [s]
            
        Returns
        -------
        dict
            TTE statistics and distribution
        """
        tte_samples = []
        
        for _ in range(n_samples):
            self.reset()
            tte = max_time
            
            while self.t < max_time:
                result = self.step(dt)
                if result['SOC'] <= SOC_cutoff:
                    tte = self.t
                    break
            
            tte_samples.append(tte)
        
        tte_samples = np.array(tte_samples)
        
        return {
            'TTE_mean': np.mean(tte_samples),
            'TTE_std': np.std(tte_samples),
            'TTE_median': np.median(tte_samples),
            'TTE_q5': np.percentile(tte_samples, 5),
            'TTE_q95': np.percentile(tte_samples, 95),
            'samples': tte_samples
        }
    
    def compute_SDE_TTE(self, P_mean, P_std, V_avg=3.7, SOC_init=1.0, SOC_cutoff=0.05):
        """
        Compute TTE using SDE formulation (Eq. 600-605).
        
        dSOC = -μ(SOC)*dt + σ(SOC)*dW
        where μ = P_mean / (Q_n * V)
              σ = P_std / (Q_n * V)
        
        Parameters
        ----------
        P_mean : float
            Mean power [W]
        P_std : float
            Power standard deviation [W]
        V_avg : float
            Average voltage [V]
        SOC_init : float
            Initial SOC
        SOC_cutoff : float
            SOC threshold
            
        Returns
        -------
        dict
            TTE mean and variance from SDE theory
        """
        Q_n = self.battery.Q_n * 3600  # Convert Ah to As
        
        # Drift and diffusion coefficients
        mu = P_mean / (Q_n * V_avg)  # [1/s]
        sigma = P_std / (Q_n * V_avg)  # [1/s]
        
        # First passage time for OU process (simplified)
        # Mean: E[TTE] ≈ (SOC_init - SOC_cutoff) / μ
        delta_SOC = SOC_init - SOC_cutoff
        TTE_mean = delta_SOC / mu if mu > 0 else np.inf
        
        # Variance approximation (for diffusion process)
        # Var[TTE] ≈ σ² * (SOC_init - SOC_cutoff)³ / (3 * μ³)
        if mu > 0 and sigma > 0:
            TTE_var = (sigma**2 * delta_SOC**3) / (3 * mu**3)
        else:
            TTE_var = 0
        
        return {
            'TTE_mean': TTE_mean,
            'TTE_std': np.sqrt(TTE_var),
            'mu': mu,
            'sigma': sigma
        }
    
    def get_state(self):
        """Get current system state."""
        return {
            'SOC': self.battery.SOC,
            'SOH': self.battery.get_SOH(),
            'V_term': self.circuit.terminal_voltage(
                self.battery.SOC, 0, self.circuit.T, self.battery.delta_SEI
            ),
            'T': self.circuit.T,
            'delta_SEI': self.battery.delta_SEI,
            't': self.t
        }
    
    def get_history_arrays(self):
        """Convert history to numpy arrays."""
        return {k: np.array(v) for k, v in self.history.items()}
