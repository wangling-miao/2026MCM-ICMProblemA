"""
Equivalent Circuit Model
========================

Implements Section 4.1.2 of the paper:
- Second-order RC Thevenin model (Eq. 435-445)
- Arrhenius temperature compensation (Eq. 452-457)
- OCV-SOC relationship (Eq. 463)
- Thermodynamic model (Eq. 472-481)
"""

import numpy as np
from .constants import R_GAS, T_REF, BatteryParams

class EquivalentCircuitModel:
    """
    Second-order RC equivalent circuit model with temperature compensation
    and thermal dynamics.
    """
    
    def __init__(self, params=None):
        """
        Initialize equivalent circuit model.
        
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
        
        # RC network state variables
        self.V1 = 0.0  # Polarization voltage 1 [V]
        self.V2 = 0.0  # Polarization voltage 2 [V]
        
        # Temperature state
        self.T = self.p.T_amb
        
        # OCV lookup (will be populated from data or polynomial)
        self.ocv_coeffs = self.p.ocv_coeffs
        
    def reset(self, T_init=None):
        """Reset model state."""
        self.V1 = 0.0
        self.V2 = 0.0
        self.T = T_init if T_init is not None else self.p.T_amb
    
    # =========================================================================
    # Arrhenius Temperature Compensation (Eq. 452-457)
    # =========================================================================
    
    def R0(self, T, delta_SEI=0.0, rho_SEI=None, A_cell=None):
        """
        Calculate ohmic resistance with temperature compensation and SEI contribution.
        
        Implements Eq. 452 + Eq. 424:
        R0(T) = R0_ref * exp[Ea(R0)/R * (1/T - 1/T_ref)] + ρ_SEI * δ_SEI / A_cell
        
        Parameters
        ----------
        T : float
            Temperature [K]
        delta_SEI : float
            SEI film thickness [m]
        rho_SEI : float, optional
            SEI resistivity [Ohm·m]
        A_cell : float, optional
            Electrode area [m²]
            
        Returns
        -------
        float
            Ohmic resistance [Ohm]
        """
        if rho_SEI is None:
            rho_SEI = self.p.rho_SEI
        if A_cell is None:
            A_cell = self.p.A_cell
            
        # Arrhenius term
        arrhenius = np.exp(self.p.Ea_R0 / R_GAS * (1.0/T - 1.0/T_REF))
        R0_base = self.p.R0_ref * arrhenius
        
        # SEI contribution
        R0_SEI = rho_SEI * delta_SEI / A_cell if A_cell > 0 else 0.0
        
        return R0_base + R0_SEI
    
    def tau1(self, T):
        """
        Calculate RC network 1 time constant with temperature compensation.
        
        Implements Eq. 456:
        τ1(T) = τ1_ref * exp[Ea(τ1)/R * (1/T - 1/T_ref)]
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Time constant τ1 [s]
        """
        arrhenius = np.exp(self.p.Ea_tau1 / R_GAS * (1.0/T - 1.0/T_REF))
        return self.p.tau1_ref * arrhenius
    
    def tau2(self, T):
        """
        Calculate RC network 2 time constant with temperature compensation.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Time constant τ2 [s]
        """
        arrhenius = np.exp(self.p.Ea_tau2 / R_GAS * (1.0/T - 1.0/T_REF))
        return self.p.tau2_ref * arrhenius
    
    def R1(self, T):
        """
        Calculate R1 resistance with temperature compensation.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Resistance R1 [Ohm]
        """
        # R1 follows similar Arrhenius behavior
        tau1_T = self.tau1(T)
        C1_T = self.C1(T)
        return tau1_T / C1_T if C1_T > 0 else self.p.R1_ref
    
    def R2(self, T):
        """
        Calculate R2 resistance with temperature compensation.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Resistance R2 [Ohm]
        """
        tau2_T = self.tau2(T)
        C2_T = self.C2(T)
        return tau2_T / C2_T if C2_T > 0 else self.p.R2_ref
    
    def C1(self, T):
        """
        Calculate C1 capacitance.
        Assumed relatively constant with temperature.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Capacitance C1 [F]
        """
        # Capacitance has weak temperature dependence
        return self.p.C1_ref
    
    def C2(self, T):
        """
        Calculate C2 capacitance.
        
        Parameters
        ----------
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            Capacitance C2 [F]
        """
        return self.p.C2_ref
    
    # =========================================================================
    # OCV-SOC Relationship (Eq. 463)
    # =========================================================================
    
    def OCV(self, SOC, T=None):
        """
        Calculate Open Circuit Voltage from SOC.
        
        Implements Eq. 463-464:
        U_ocv(SOC, T) = Σ a_k(T) * SOC^k
        
        Parameters
        ----------
        SOC : float or array
            State of charge [0-1]
        T : float, optional
            Temperature [K]. If None, uses reference temperature.
            
        Returns
        -------
        float or array
            Open circuit voltage [V]
        """
        if T is None:
            T = T_REF
        
        SOC = np.clip(SOC, 0.0, 1.0)
        
        # Polynomial evaluation
        # ocv_coeffs can be in either format - check and handle
        coeffs = np.array(self.ocv_coeffs).flatten()
        
        # np.polyval expects coeffs in descending order [a_n, a_{n-1}, ..., a_0]
        # If coeffs look like they're in ascending order, reverse them
        if len(coeffs) >= 2:
            # Check: evaluate at SOC=0 should give ~3V, at SOC=1 should give ~4.2V
            test_0 = np.polyval(coeffs, 0)
            test_1 = np.polyval(coeffs, 1)
            if not (2.5 < test_0 < 4.5 and 3.0 < test_1 < 4.5):
                # Try reversed
                test_0_rev = np.polyval(coeffs[::-1], 0)
                test_1_rev = np.polyval(coeffs[::-1], 1)
                if 2.5 < test_0_rev < 4.5 and 3.0 < test_1_rev < 4.5:
                    coeffs = coeffs[::-1]
        
        ocv = np.polyval(coeffs, SOC)
        
        # Ensure reasonable voltage range
        ocv = np.clip(ocv, 2.5, 4.5)
        
        # Temperature correction (linear approximation)
        dT = T - T_REF
        ocv += self.p.dOCV_dT * dT
        
        return ocv
    
    def dOCV_dSOC(self, SOC):
        """
        Calculate derivative of OCV with respect to SOC.
        
        Parameters
        ----------
        SOC : float
            State of charge [0-1]
            
        Returns
        -------
        float
            dOCV/dSOC [V]
        """
        # Derivative of polynomial
        coeffs_deriv = np.polyder(self.ocv_coeffs[::-1])
        return np.polyval(coeffs_deriv, SOC)
    
    def set_ocv_lookup(self, soc_values, ocv_values):
        """
        Set OCV-SOC lookup table from measured data.
        
        Parameters
        ----------
        soc_values : array
            SOC values [0-1]
        ocv_values : array
            Corresponding OCV values [V]
        """
        # Fit polynomial to data
        self.ocv_coeffs = np.polyfit(soc_values, ocv_values, deg=7)
        self.ocv_coeffs = self.ocv_coeffs[::-1]  # Reverse for polyval compatibility
    
    # =========================================================================
    # RC Network Dynamics (Eq. 442-444)
    # =========================================================================
    
    def dV1_dt(self, V1, I, T):
        """
        Calculate rate of change of V1.
        
        Implements Eq. 442:
        dV1/dt = -V1/τ1 + I/C1
        
        Parameters
        ----------
        V1 : float
            Current V1 value [V]
        I : float
            Current [A]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            dV1/dt [V/s]
        """
        tau1_T = self.tau1(T)
        C1_T = self.C1(T)
        
        return -V1 / tau1_T + I / C1_T
    
    def dV2_dt(self, V2, I, T):
        """
        Calculate rate of change of V2.
        
        Implements Eq. 442:
        dV2/dt = -V2/τ2 + I/C2
        
        Parameters
        ----------
        V2 : float
            Current V2 value [V]
        I : float
            Current [A]
        T : float
            Temperature [K]
            
        Returns
        -------
        float
            dV2/dt [V/s]
        """
        tau2_T = self.tau2(T)
        C2_T = self.C2(T)
        
        return -V2 / tau2_T + I / C2_T
    
    def update_polarization(self, I, T, dt):
        """
        Update polarization voltages V1 and V2 using Euler method.
        
        Parameters
        ----------
        I : float
            Current [A]
        T : float
            Temperature [K]
        dt : float
            Time step [s]
        """
        self.V1 += self.dV1_dt(self.V1, I, T) * dt
        self.V2 += self.dV2_dt(self.V2, I, T) * dt
    
    def update_polarization_exact(self, I, T, dt):
        """
        Update polarization voltages using exact discrete solution.
        
        For constant current over dt:
        V_i(t+dt) = V_i(t)*exp(-dt/τ_i) + R_i*I*(1 - exp(-dt/τ_i))
        
        Parameters
        ----------
        I : float
            Current [A]
        T : float
            Temperature [K]
        dt : float
            Time step [s]
        """
        tau1_T = self.tau1(T)
        tau2_T = self.tau2(T)
        R1_T = self.R1(T)
        R2_T = self.R2(T)
        
        exp1 = np.exp(-dt / tau1_T)
        exp2 = np.exp(-dt / tau2_T)
        
        self.V1 = self.V1 * exp1 + R1_T * I * (1.0 - exp1)
        self.V2 = self.V2 * exp2 + R2_T * I * (1.0 - exp2)
    
    # =========================================================================
    # Terminal Voltage (Eq. 435)
    # =========================================================================
    
    def terminal_voltage(self, SOC, I, T, delta_SEI=0.0):
        """
        Calculate terminal voltage.
        
        Implements Eq. 435:
        V_term = U_ocv(SOC, T) - I*R0(T) - V1 - V2
        
        Parameters
        ----------
        SOC : float
            State of charge [0-1]
        I : float
            Current [A], positive for discharge
        T : float
            Temperature [K]
        delta_SEI : float
            SEI film thickness [m]
            
        Returns
        -------
        float
            Terminal voltage [V]
        """
        U_ocv = self.OCV(SOC, T)
        R0_T = self.R0(T, delta_SEI)
        
        V_term = U_ocv - I * R0_T - self.V1 - self.V2
        
        return V_term
    
    # =========================================================================
    # Thermal Dynamics (Eq. 472-481)
    # =========================================================================
    
    def heat_generation(self, I, T, SOC):
        """
        Calculate heat generation power.
        
        Implements Eq. 479:
        P_heat = I²*R0 + I*T*(∂U_ocv/∂T)
        
        Parameters
        ----------
        I : float
            Current [A]
        T : float
            Temperature [K]
        SOC : float
            State of charge [0-1]
            
        Returns
        -------
        float
            Heat generation power [W]
        """
        R0_T = self.R0(T)
        
        # Ohmic heat (Joule heating)
        P_ohmic = I ** 2 * R0_T
        
        # Entropic heat (reversible heat)
        # ∂U_ocv/∂T is approximately constant (dOCV_dT)
        P_entropic = I * T * self.p.dOCV_dT
        
        # Polarization heat (optional, usually small)
        R1_T = self.R1(T)
        R2_T = self.R2(T)
        P_polarization = (self.V1 ** 2) / R1_T + (self.V2 ** 2) / R2_T
        
        return P_ohmic + P_entropic + P_polarization
    
    def dT_dt(self, I, SOC, T_amb=None):
        """
        Calculate temperature rate of change.
        
        Implements Eq. 472:
        Cp*m*(dT/dt) = P_heat - h*A*(T - T_amb)
        
        Parameters
        ----------
        I : float
            Current [A]
        SOC : float
            State of charge [0-1]
        T_amb : float, optional
            Ambient temperature [K]
            
        Returns
        -------
        float
            dT/dt [K/s]
        """
        if T_amb is None:
            T_amb = self.p.T_amb
        
        P_heat = self.heat_generation(I, self.T, SOC)
        P_dissipation = self.p.h_conv * self.p.A_surface * (self.T - T_amb)
        
        thermal_mass = self.p.Cp * self.p.mass
        
        return (P_heat - P_dissipation) / thermal_mass
    
    def update_temperature(self, I, SOC, T_amb, dt):
        """
        Update temperature using Euler integration.
        
        Parameters
        ----------
        I : float
            Current [A]
        SOC : float
            State of charge [0-1]
        T_amb : float
            Ambient temperature [K]
        dt : float
            Time step [s]
            
        Returns
        -------
        float
            Updated temperature [K]
        """
        dT = self.dT_dt(I, SOC, T_amb) * dt
        self.T += dT
        
        # Physical limits
        self.T = np.clip(self.T, 233.15, 353.15)  # -40°C to 80°C
        
        return self.T
    
    # =========================================================================
    # Complete State Update
    # =========================================================================
    
    def step(self, SOC, I, dt, T_amb=None, delta_SEI=0.0, use_exact=True):
        """
        Perform one time step of circuit model update.
        
        Parameters
        ----------
        SOC : float
            State of charge [0-1]
        I : float
            Current [A]
        dt : float
            Time step [s]
        T_amb : float, optional
            Ambient temperature [K]
        delta_SEI : float
            SEI film thickness [m]
        use_exact : bool
            Use exact discrete solution for polarization
            
        Returns
        -------
        dict
            Updated state and outputs
        """
        if T_amb is None:
            T_amb = self.p.T_amb
        
        # Update temperature
        self.update_temperature(I, SOC, T_amb, dt)
        
        # Update polarization voltages
        if use_exact:
            self.update_polarization_exact(I, self.T, dt)
        else:
            self.update_polarization(I, self.T, dt)
        
        # Calculate terminal voltage
        V_term = self.terminal_voltage(SOC, I, self.T, delta_SEI)
        
        # Heat generation
        P_heat = self.heat_generation(I, self.T, SOC)
        
        return {
            'V_term': V_term,
            'V1': self.V1,
            'V2': self.V2,
            'T': self.T,
            'R0': self.R0(self.T, delta_SEI),
            'OCV': self.OCV(SOC, self.T),
            'P_heat': P_heat
        }
    
    # =========================================================================
    # State Vector Interface (for EKF)
    # =========================================================================
    
    def get_state_vector(self, SOC):
        """
        Get state vector [SOC, V1, V2]^T for EKF.
        
        Parameters
        ----------
        SOC : float
            State of charge
            
        Returns
        -------
        ndarray
            State vector (3,)
        """
        return np.array([SOC, self.V1, self.V2])
    
    def set_state_vector(self, x, SOC_external=None):
        """
        Set state from vector.
        
        Parameters
        ----------
        x : ndarray
            State vector [SOC, V1, V2]
        SOC_external : float, optional
            External SOC reference (not stored here)
        """
        self.V1 = x[1]
        self.V2 = x[2]
