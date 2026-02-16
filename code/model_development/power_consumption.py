"""
Power Consumption Model
=======================

Implements Section 4.2 of the paper:
- Multi-dimensional power decomposition (Eq. 490)
- Display power model (Eq. 496)
- CPU power with thermal throttling (Eq. 504-517)
- Wireless connectivity power (Eq. 522-528)
- Background power (Eq. 533)
- Edge AI power for heterogeneous multi-core (Eq. 541)
"""

import numpy as np
from .constants import SmartphoneParams, T_REF

class PowerConsumptionModel:
    """
    Multi-component smartphone power consumption model with 
    thermal throttling feedback.
    """
    
    def __init__(self, params=None):
        """
        Initialize power consumption model.
        
        Parameters
        ----------
        params : SmartphoneParams or dict, optional
            Smartphone parameters. If None, uses default SmartphoneParams.
        """
        if params is None:
            self.p = SmartphoneParams()
        elif isinstance(params, dict):
            self.p = SmartphoneParams()
            for key, value in params.items():
                if hasattr(self.p, key):
                    setattr(self.p, key, value)
        else:
            self.p = params
        
        # Current state
        self.gamma = 1.0  # Throttling factor
        self.current_mode = 'Light'
        
    def reset(self):
        """Reset power model state."""
        self.gamma = 1.0
        self.current_mode = 'Light'
    
    # =========================================================================
    # Display Power (Eq. 496-498)
    # =========================================================================
    
    def display_power(self, brightness, I_red=0.5, I_green=0.5, I_blue=0.5):
        """
        Calculate OLED display power consumption.
        
        Implements Eq. 496:
        P_display = P_base + η_b*b + Σ η_c*Ī_c
        
        Parameters
        ----------
        brightness : float
            Normalized brightness [0-1]
        I_red : float
            Average red pixel intensity [0-1]
        I_green : float
            Average green pixel intensity [0-1]
        I_blue : float
            Average blue pixel intensity [0-1]
            
        Returns
        -------
        float
            Display power [W]
        """
        P_brightness = self.p.eta_brightness * brightness
        P_color = (self.p.eta_red * I_red + 
                   self.p.eta_green * I_green + 
                   self.p.eta_blue * I_blue)
        
        return self.p.P_base_display + P_brightness + P_color
    
    # =========================================================================
    # Thermal Throttling (Eq. 505-511)
    # =========================================================================
    
    def throttling_factor(self, T, V_term):
        """
        Calculate dynamic frequency scaling factor γ(t).
        
        Implements Eq. 510:
        γ = min(1, max(0, (T_limit - T)/(T_limit - T_thresh)), 
                   max(0, (V_term - V_cutoff)/ΔV_safe))
        
        Parameters
        ----------
        T : float
            Battery temperature [K]
        V_term : float
            Terminal voltage [V]
            
        Returns
        -------
        float
            Throttling factor γ [0-1]
        """
        # Temperature-based throttling
        if T >= self.p.T_limit:
            gamma_T = 0.0
        elif T <= self.p.T_thresh:
            gamma_T = 1.0
        else:
            gamma_T = (self.p.T_limit - T) / (self.p.T_limit - self.p.T_thresh)
        
        # Voltage-based throttling
        if V_term <= self.p.V_cutoff:
            gamma_V = 0.0
        else:
            gamma_V = (V_term - self.p.V_cutoff) / self.p.dV_safe
            gamma_V = min(1.0, gamma_V)
        
        # Combined throttling (take minimum)
        self.gamma = min(gamma_T, gamma_V)
        self.gamma = max(0.0, min(1.0, self.gamma))
        
        return self.gamma
    
    # =========================================================================
    # CPU Power (Eq. 515-517)
    # =========================================================================
    
    def cpu_power(self, f_req, T, V_term, V_dd=None):
        """
        Calculate CPU power with DVFS and thermal throttling.
        
        Implements Eq. 515:
        P_CPU = C_eff * V_dd² * (f_req * γ) + P_leak(T)
        
        Parameters
        ----------
        f_req : float
            Requested frequency [Hz]
        T : float
            Temperature [K]
        V_term : float
            Battery terminal voltage [V]
        V_dd : float, optional
            Supply voltage [V]. If None, derived from frequency.
            
        Returns
        -------
        float
            CPU power [W]
        """
        # Update throttling factor
        gamma = self.throttling_factor(T, V_term)
        
        # Actual frequency
        f_actual = f_req * gamma
        
        # Voltage scaling (linear approximation with frequency)
        if V_dd is None:
            # V_dd scales with frequency
            f_norm = f_actual / self.p.f_max
            V_dd = self.p.V_dd_min + (self.p.V_dd_max - self.p.V_dd_min) * f_norm
        
        # Dynamic power
        P_dynamic = self.p.C_eff * (V_dd ** 2) * f_actual
        
        # Leakage power (exponential with temperature)
        dT = T - T_REF
        P_leak = self.p.P_leak_ref * np.exp(self.p.leak_temp_coeff * dT)
        
        return P_dynamic + P_leak
    
    # =========================================================================
    # Wireless Connectivity Power (Eq. 522-528)
    # =========================================================================
    
    def connectivity_power(self, mode='WiFi', signal_strength=1.0):
        """
        Calculate wireless connectivity power.
        
        Implements Eq. 522:
        P_conn depends on connection type and signal strength
        
        Parameters
        ----------
        mode : str
            Connection mode: 'WiFi', '4G', '5G', or 'Off'
        signal_strength : float
            Normalized signal strength [0-1] (1 = excellent)
            
        Returns
        -------
        float
            Connectivity power [W]
        """
        if mode == 'Off':
            return 0.0
        elif mode == 'WiFi':
            return self.p.P_WiFi
        elif mode == '4G':
            # Weak signal increases power
            signal_factor = 1.0 + self.p.alpha_signal_4G * (1.0 - signal_strength)
            return self.p.P_4G * signal_factor
        elif mode == '5G':
            signal_factor = 1.0 + self.p.beta_signal_5G * (1.0 - signal_strength)
            return self.p.P_5G * signal_factor
        else:
            return self.p.P_WiFi  # Default to WiFi
    
    # =========================================================================
    # Background Power (Eq. 533-535)
    # =========================================================================
    
    def background_power(self, n_apps=5):
        """
        Calculate background power consumption.
        
        Implements Eq. 533:
        P_bg = P_idle + k_bg * N_app
        
        Parameters
        ----------
        n_apps : int
            Number of active background applications
            
        Returns
        -------
        float
            Background power [W]
        """
        return self.p.P_idle + self.p.k_bg * n_apps
    
    # =========================================================================
    # Edge AI Power (Eq. 541)
    # =========================================================================
    
    def ai_power(self, n_little=0, n_medium=0, n_big=0, 
                 N_little=4, N_medium=2, N_big=2, 
                 freq_util=0.8):
        """
        Calculate AI workload power for heterogeneous multi-core processor.
        
        Implements Eq. 541:
        P_AI(S) = P_static + Σ α_j * [n_j + (N_j - n_j)*β] * (f_max,j * u)²
        
        Parameters
        ----------
        n_little : int
            Number of Little cores executing AI tasks
        n_medium : int
            Number of Medium cores executing AI tasks
        n_big : int
            Number of Big cores executing AI tasks
        N_little : int
            Total number of Little cores
        N_medium : int
            Total number of Medium cores
        N_big : int
            Total number of Big cores
        freq_util : float
            Frequency utilization rate [0-1]
            
        Returns
        -------
        float
            AI power [W]
        """
        P = self.p.P_static_AI
        beta = self.p.beta_idle_core
        
        # Frequency squared (normalized)
        f_sq = freq_util ** 2
        
        # Little cores contribution
        n_active_little = n_little + (N_little - n_little) * beta
        P += self.p.alpha_little * n_active_little * (self.p.f_max * f_sq) ** 2
        
        # Medium cores contribution (assuming f_max,medium = 0.8 * f_max)
        f_medium = 0.8 * self.p.f_max
        n_active_medium = n_medium + (N_medium - n_medium) * beta
        P += self.p.alpha_medium * n_active_medium * (f_medium * f_sq) ** 2
        
        # Big cores contribution
        n_active_big = n_big + (N_big - n_big) * beta
        P += self.p.alpha_big * n_active_big * (self.p.f_max * f_sq) ** 2
        
        return P
    
    # =========================================================================
    # Total Power (Eq. 490)
    # =========================================================================
    
    def total_power(self, brightness=0.5, f_req=None, T=298.15, V_term=3.8,
                    conn_mode='WiFi', signal_strength=1.0, n_bg_apps=5,
                    ai_active=False, n_big_cores=0, freq_util=0.5,
                    I_red=0.5, I_green=0.5, I_blue=0.5):
        """
        Calculate total power consumption.
        
        Implements Eq. 490:
        P_total = P_display + P_CPU + P_conn + P_bg + P_AI
        
        Parameters
        ----------
        brightness : float
            Screen brightness [0-1]
        f_req : float
            Requested CPU frequency [Hz]
        T : float
            Temperature [K]
        V_term : float
            Terminal voltage [V]
        conn_mode : str
            Connection mode
        signal_strength : float
            Signal strength [0-1]
        n_bg_apps : int
            Number of background apps
        ai_active : bool
            Whether AI workload is active
        n_big_cores : int
            Number of big cores for AI
        freq_util : float
            Frequency utilization
        I_red, I_green, I_blue : float
            Pixel intensities [0-1]
            
        Returns
        -------
        float
            Total power [W]
        """
        if f_req is None:
            f_req = self.p.f_max * 0.5  # Default 50% frequency
        
        # Display
        P_display = self.display_power(brightness, I_red, I_green, I_blue)
        
        # CPU
        P_CPU = self.cpu_power(f_req, T, V_term)
        
        # Connectivity
        P_conn = self.connectivity_power(conn_mode, signal_strength)
        
        # Background
        P_bg = self.background_power(n_bg_apps)
        
        # AI (if active)
        if ai_active:
            P_AI = self.ai_power(n_big=n_big_cores, freq_util=freq_util)
        else:
            P_AI = 0.0
        
        return P_display + P_CPU + P_conn + P_bg + P_AI
    
    # =========================================================================
    # Mode-based Power (Eq. 576-577)
    # =========================================================================
    
    def mode_power(self, mode='Light', noise_std=0.1):
        """
        Get power consumption for a usage mode with noise.
        
        Implements Eq. 575-578:
        P_total(t) = P_m(t) + ξ(t)
        
        Parameters
        ----------
        mode : str
            Usage mode: 'Idle', 'Light', 'Busy', 'Gaming', 'Video', 'Navigation'
        noise_std : float
            Standard deviation of power fluctuation noise [W]
            
        Returns
        -------
        float
            Power with noise [W]
        """
        self.current_mode = mode
        P_base = self.p.mode_powers.get(mode, self.p.mode_powers['Light'])
        
        # Add zero-mean Gaussian noise
        noise = np.random.normal(0, noise_std)
        
        return max(0.1, P_base + noise)  # Minimum 0.1W
    
    # =========================================================================
    # Power to Current Conversion
    # =========================================================================
    
    def power_to_current(self, P_total, V_term):
        """
        Convert power to discharge current.
        
        I = P / V
        
        Parameters
        ----------
        P_total : float
            Total power [W]
        V_term : float
            Terminal voltage [V]
            
        Returns
        -------
        float
            Discharge current [A]
        """
        if V_term <= 0:
            return 0.0
        return P_total / V_term
    
    # =========================================================================
    # Detailed Component Breakdown
    # =========================================================================
    
    def get_power_breakdown(self, brightness=0.5, f_req=None, T=298.15, 
                            V_term=3.8, conn_mode='WiFi', signal_strength=1.0,
                            n_bg_apps=5, ai_active=False, n_big_cores=0,
                            freq_util=0.5):
        """
        Get detailed power breakdown by component.
        
        Parameters
        ----------
        (Same as total_power)
        
        Returns
        -------
        dict
            Power breakdown by component
        """
        if f_req is None:
            f_req = self.p.f_max * 0.5
        
        P_display = self.display_power(brightness)
        P_CPU = self.cpu_power(f_req, T, V_term)
        P_conn = self.connectivity_power(conn_mode, signal_strength)
        P_bg = self.background_power(n_bg_apps)
        P_AI = self.ai_power(n_big=n_big_cores, freq_util=freq_util) if ai_active else 0.0
        
        P_total = P_display + P_CPU + P_conn + P_bg + P_AI
        
        return {
            'P_display': P_display,
            'P_CPU': P_CPU,
            'P_conn': P_conn,
            'P_bg': P_bg,
            'P_AI': P_AI,
            'P_total': P_total,
            'throttling_factor': self.gamma
        }
