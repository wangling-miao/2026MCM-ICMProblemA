"""
Battery State Dynamics Model
============================

Implements Section 4.1 of the paper:
- SOC differential equation (Eq. 356-357)
- Self-discharge model (Eq. 363)
- Capacity fade model (Eq. 378-403)
- SEI film growth kinetics (Eq. 416-426)
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from .constants import R_GAS, F_FARADAY, T_REF, BatteryParams

class BatteryDynamics:
    """
    Implements the battery state dynamics including SOC evolution,
    capacity fade, and aging mechanisms.
    """
    
    def __init__(self, params=None):
        """
        Initialize battery dynamics model.
        
        Parameters
        ----------
        params : BatteryParams or dict, optional
            Battery parameters. If None, uses default BatteryParams.
        """
        if params is None:
            self.p = BatteryParams()
        elif isinstance(params, dict):
            self.p = BatteryParams()
            for key, value in params.items():
                if hasattr(self.p, key):
                    setattr(self.p, key, value)
        else:
            self.p = params
        
        # State variables
        self.SOC = 1.0
        self.Q_n = self.p.Q_n0
        self.delta_SEI = 0.0
        self.N_eq = 0.0
        self.t_total = 0.0
        
        # History for rainflow counting
        self.SOC_history = []
        
    def reset(self, SOC_init=1.0):
        """Reset battery state to initial conditions."""
        self.SOC = SOC_init
        self.Q_n = self.p.Q_n0
        self.delta_SEI = 0.0
        self.N_eq = 0.0
        self.t_total = 0.0
        self.SOC_history = [SOC_init]
        
    # =========================================================================
    # SOC Dynamics (Eq. 356-357)
    # =========================================================================
    
    def self_discharge_rate(self, T, SOC):
        """
        Calculate self-discharge rate λ_leak(T, SOC).
        
        Implements Eq. 363:
        λ_leak = k0 * SOC * exp(-Ea_leak / (R*T))
        
        Parameters
        ----------
        T : float
            Temperature [K]
        SOC : float
            State of charge [0-1]
            
        Returns
        -------
        float
            Self-discharge rate [1/s]
        """
        return self.p.k0_leak * SOC * np.exp(-self.p.Ea_leak / (R_GAS * T))
    
    def dSOC_dt(self, I, T, SOC=None):
        """
        Calculate SOC rate of change.
        
        Implements Eq. 356:
        dSOC/dt = -I(t)/Q_n(t) - λ_leak(T, SOC)
        
        Parameters
        ----------
        I : float
            Current [A], positive for discharge
        T : float
            Temperature [K]
        SOC : float, optional
            State of charge. If None, uses internal state.
            
        Returns
        -------
        float
            Rate of change of SOC [1/s]
        """
        if SOC is None:
            SOC = self.SOC
            
        # Coulomb counting term
        # Q_n is in Ah, need to convert to As for time in seconds
        Q_n_As = self.Q_n * 3600.0  # Convert Ah to As (Coulombs)
        coulomb_term = -I / Q_n_As
        
        # Self-discharge term
        leak_term = -self.self_discharge_rate(T, SOC)
        
        return coulomb_term + leak_term
    
    def update_SOC(self, I, T, dt):
        """
        Update SOC using Euler integration.
        
        Parameters
        ----------
        I : float
            Current [A]
        T : float
            Temperature [K]
        dt : float
            Time step [s]
            
        Returns
        -------
        float
            Updated SOC
        """
        dSOC = self.dSOC_dt(I, T) * dt
        self.SOC = np.clip(self.SOC + dSOC, 0.0, 1.0)
        self.SOC_history.append(self.SOC)
        return self.SOC
    
    # =========================================================================
    # Capacity Fade Model (Eq. 378-403)
    # =========================================================================
    
    def soc_stress_function(self, SOC_avg):
        """
        SOC-dependent stress function g(SOC).
        
        Implements Eq. 390:
        g(SOC) = 1 + β_soc * (SOC - 0.5)²
        
        Parameters
        ----------
        SOC_avg : float
            Average SOC during storage
            
        Returns
        -------
        float
            SOC stress factor
        """
        return 1.0 + self.p.beta_soc * (SOC_avg - 0.5) ** 2
    
    def crate_stress_function(self, C_rate):
        """
        C-rate stress function h(C_rate).
        
        Implements Eq. 401:
        h(C_rate) = 1 + δ_c * (C_rate - 1)²
        
        Parameters
        ----------
        C_rate : float
            Charge/discharge rate
            
        Returns
        -------
        float
            C-rate stress factor
        """
        return 1.0 + self.p.delta_crate * (C_rate - 1.0) ** 2
    
    def calendar_aging(self, t, T, SOC_avg=0.5):
        """
        Calculate calendar aging loss L_cal.
        
        Implements Eq. 385:
        L_cal = α_cal * √t * exp(-Ea_cal/(R*T)) * g(SOC_avg)
        
        Parameters
        ----------
        t : float
            Total time [s]
        T : float
            Temperature [K]
        SOC_avg : float
            Average SOC during storage
            
        Returns
        -------
        float
            Calendar aging loss fraction
        """
        arrhenius = np.exp(-self.p.Ea_cal / (R_GAS * T))
        soc_stress = self.soc_stress_function(SOC_avg)
        
        return self.p.alpha_cal * np.sqrt(t) * arrhenius * soc_stress
    
    def cycle_aging(self, N_eq, T, C_rate=1.0):
        """
        Calculate cycle aging loss L_cyc.
        
        Implements Eq. 396:
        L_cyc = α_cyc * N_eq^0.5 * exp(γ*(T - T_ref)) * h(C_rate)
        
        Parameters
        ----------
        N_eq : float
            Equivalent cycle number
        T : float
            Temperature [K]
        C_rate : float
            Average C-rate
            
        Returns
        -------
        float
            Cycle aging loss fraction
        """
        temp_factor = np.exp(self.p.gamma_temp * (T - T_REF))
        crate_stress = self.crate_stress_function(C_rate)
        
        return self.p.alpha_cyc * np.sqrt(N_eq) * temp_factor * crate_stress
    
    def rainflow_counting(self, SOC_trajectory):
        """
        Extract equivalent cycle number using simplified rainflow counting.
        
        Implements Eq. 407:
        N_eq = Σ (ΔDOD_i / 2)
        
        Parameters
        ----------
        SOC_trajectory : array-like
            SOC values over time
            
        Returns
        -------
        float
            Equivalent cycle number
        """
        if len(SOC_trajectory) < 2:
            return 0.0
        
        # Find local extrema
        extrema = []
        for i in range(1, len(SOC_trajectory) - 1):
            if ((SOC_trajectory[i] > SOC_trajectory[i-1] and 
                 SOC_trajectory[i] > SOC_trajectory[i+1]) or
                (SOC_trajectory[i] < SOC_trajectory[i-1] and 
                 SOC_trajectory[i] < SOC_trajectory[i+1])):
                extrema.append(SOC_trajectory[i])
        
        if len(extrema) < 2:
            # Use overall range as single half-cycle
            DOD = abs(max(SOC_trajectory) - min(SOC_trajectory))
            return DOD / 2.0
        
        # Simplified rainflow: sum of half-cycle DODs
        N_eq = 0.0
        for i in range(len(extrema) - 1):
            delta_DOD = abs(extrema[i+1] - extrema[i])
            N_eq += delta_DOD / 2.0
        
        return N_eq
    
    def update_equivalent_cycles(self):
        """Update equivalent cycle count from SOC history."""
        self.N_eq = self.rainflow_counting(self.SOC_history)
        return self.N_eq
    
    # =========================================================================
    # SEI Film Growth Kinetics (Eq. 416-426)
    # =========================================================================
    
    def sei_growth_rate(self, T, eta_side):
        """
        Calculate SEI film growth rate.
        
        Implements Eq. 417:
        dδ_SEI/dt = K_SEI * exp(-Ea_SEI/(R*T)) * exp(-αF/(R*T) * η_side)
        
        Parameters
        ----------
        T : float
            Temperature [K]
        eta_side : float
            Side reaction overpotential [V]
            
        Returns
        -------
        float
            SEI growth rate [m/s]
        """
        arrhenius_term = np.exp(-self.p.Ea_SEI / (R_GAS * T))
        tafel_term = np.exp(-self.p.alpha_SEI * F_FARADAY / (R_GAS * T) * eta_side)
        
        return self.p.K_SEI * arrhenius_term * tafel_term
    
    def sei_resistance_contribution(self, delta_SEI):
        """
        Calculate resistance increase due to SEI film.
        
        Implements Eq. 424:
        ΔR0 = ρ_SEI * δ_SEI / A_cell
        
        Parameters
        ----------
        delta_SEI : float
            SEI film thickness [m]
            
        Returns
        -------
        float
            Resistance increase [Ohm]
        """
        return self.p.rho_SEI * delta_SEI / self.p.A_cell
    
    def sei_capacity_loss(self, delta_SEI):
        """
        Calculate capacity loss due to SEI lithium consumption.
        
        Implements Eq. 425:
        ΔQ = λ_loss * δ_SEI
        
        Parameters
        ----------
        delta_SEI : float
            SEI film thickness [m]
            
        Returns
        -------
        float
            Capacity loss [Ah]
        """
        return self.p.lambda_loss * delta_SEI
    
    def update_SEI(self, T, V_term, dt):
        """
        Update SEI film thickness.
        
        Parameters
        ----------
        T : float
            Temperature [K]
        V_term : float
            Terminal voltage [V] (used as proxy for side reaction potential)
        dt : float
            Time step [s]
            
        Returns
        -------
        float
            Updated SEI thickness [m]
        """
        # Side reaction overpotential approximation
        # Higher voltage leads to higher side reaction rate
        eta_side = max(0, V_term - 3.7)  # Threshold at ~3.7V
        
        growth_rate = self.sei_growth_rate(T, eta_side)
        self.delta_SEI += growth_rate * dt
        
        return self.delta_SEI
    
    # =========================================================================
    # Capacity Update (Eq. 379, 425)
    # =========================================================================
    
    def update_capacity(self, t_total, T, C_rate=1.0, SOC_avg=0.5):
        """
        Update effective capacity considering all aging mechanisms.
        
        Implements Eq. 379 combined with SEI loss (Eq. 425):
        Q_n = Q_n0 * (1 - L_cal - L_cyc) - λ_loss * δ_SEI
        
        Parameters
        ----------
        t_total : float
            Total operation time [s]
        T : float
            Temperature [K]
        C_rate : float
            Average C-rate
        SOC_avg : float
            Average SOC during storage periods
            
        Returns
        -------
        float
            Updated effective capacity [Ah]
        """
        # Update equivalent cycles from history
        self.update_equivalent_cycles()
        
        # Calendar and cycle aging losses
        L_cal = self.calendar_aging(t_total, T, SOC_avg)
        L_cyc = self.cycle_aging(self.N_eq, T, C_rate)
        
        # Capacity from cycle/calendar aging
        Q_aged = self.p.Q_n0 * (1.0 - L_cal - L_cyc)
        
        # Additional loss from SEI
        Q_SEI_loss = self.sei_capacity_loss(self.delta_SEI)
        
        # Total effective capacity
        self.Q_n = max(0.5, Q_aged - Q_SEI_loss)  # Minimum 0.5 Ah
        
        return self.Q_n
    
    def get_SOH(self):
        """
        Get State of Health.
        
        Implements Eq. 372:
        SOH = Q_n / Q_n0
        
        Returns
        -------
        float
            State of health [0-1]
        """
        return self.Q_n / self.p.Q_n0
    
    # =========================================================================
    # Full State Update
    # =========================================================================
    
    def step(self, I, T, V_term, dt):
        """
        Perform one time step of battery dynamics update.
        
        Parameters
        ----------
        I : float
            Current [A]
        T : float
            Temperature [K]
        V_term : float
            Terminal voltage [V]
        dt : float
            Time step [s]
            
        Returns
        -------
        dict
            Updated state variables
        """
        self.t_total += dt
        
        # Update SOC
        self.update_SOC(I, T, dt)
        
        # Update SEI
        self.update_SEI(T, V_term, dt)
        
        # Update capacity (less frequently for efficiency)
        if int(self.t_total) % 3600 == 0:  # Every hour
            C_rate = abs(I) / self.p.Q_n0
            self.update_capacity(self.t_total, T, C_rate)
        
        return {
            'SOC': self.SOC,
            'Q_n': self.Q_n,
            'SOH': self.get_SOH(),
            'delta_SEI': self.delta_SEI,
            'N_eq': self.N_eq
        }
