"""
Physical Constants and Default Parameters
==========================================

Contains all physical constants and default battery/smartphone parameters
used throughout the model development.

Updated to match the identified values in the paper (Table 1 in Section 5.2).
Equations reference updated to match new.tex numbering.
"""

import numpy as np

# =============================================================================
# Universal Physical Constants
# =============================================================================
R_GAS = 8.314          # Universal gas constant [J/(mol·K)]
F_FARADAY = 96485.0    # Faraday constant [C/mol]
K_BOLTZMANN = 1.38e-23 # Boltzmann constant [J/K]

# =============================================================================
# Reference Conditions
# =============================================================================
T_REF = 298.15         # Reference temperature [K] (25°C)
T_REF_C = 25.0         # Reference temperature [°C]

# =============================================================================
# Default Battery Parameters (Typical Smartphone Li-ion Cell)
# Updated based on CALCE dataset parameter identification (Section 5.2)
# =============================================================================
class BatteryParams:
    """Default parameters for a typical smartphone lithium-ion battery.
    
    Parameter values from Table 1 (基于实验数据识别的关键模型参数):
    - R0_ref = 29.11 mΩ (0.02911 Ω)
    - Ea(R0) = 15,162 J/mol
    - Q_n0 = 3.0 Ah
    - OCV RMSE = 9.82 mV
    """
    
    # Capacity (Paper Table 1)
    Q_n0 = 3.0             # Initial nominal capacity [Ah] - from paper
    V_cutoff = 3.0         # Cutoff voltage [V]
    V_max = 4.35           # Maximum voltage [V]
    
    # Ohmic Resistance (Paper Table 1: R0_ref = 29.11 mΩ)
    R0_ref = 0.02911       # Reference ohmic resistance at T_ref [Ohm] - from paper
    Ea_R0 = 15162.0        # Activation energy for R0 [J/mol] - from paper (15,162 J/mol)
    
    # RC Network 1 (Electrochemical Polarization)
    R1_ref = 0.015         # Reference R1 at T_ref [Ohm]
    C1_ref = 2500.0        # Reference C1 at T_ref [F]
    tau1_ref = 37.5        # Reference time constant tau1 = R1*C1 [s]
    Ea_tau1 = 12000.0      # Activation energy for tau1 [J/mol]
    
    # RC Network 2 (Concentration Polarization)
    R2_ref = 0.008         # Reference R2 at T_ref [Ohm]
    C2_ref = 12500.0       # Reference C2 at T_ref [F]
    tau2_ref = 100.0       # Reference time constant tau2 = R2*C2 [s]
    Ea_tau2 = 15000.0      # Activation energy for tau2 [J/mol]
    
    # Self-discharge
    k0_leak = 1e-8         # Self-discharge rate constant [1/s]
    Ea_leak = 50000.0      # Activation energy for self-discharge [J/mol]
    
    # Thermal Parameters (Eq. thermal_dynamics in Section 4.3.3)
    Cp = 1000.0            # Specific heat capacity [J/(kg·K)]
    mass = 0.045           # Battery mass [kg]
    h_conv = 10.0          # Convective heat transfer coefficient [W/(m²·K)]
    A_surface = 0.005      # Battery surface area [m²]
    T_amb = 298.15         # Ambient temperature [K]
    
    # Aging Parameters - Calendar (Eq. capacity_fade)
    alpha_cal = 5e-5       # Calendar aging coefficient [s^-0.5]
    Ea_cal = 50000.0       # Activation energy for calendar aging [J/mol]
    beta_soc = 0.5         # SOC stress coefficient
    
    # Aging Parameters - Cycle
    alpha_cyc = 1e-4       # Cycle aging coefficient
    gamma_temp = 0.02      # Temperature sensitivity coefficient [1/K]
    delta_crate = 0.1      # C-rate stress coefficient
    
    # SEI Film Growth (Eq. sei_growth, sei_current)
    K_SEI = 1e-12          # SEI growth rate constant [m/s]
    Ea_SEI = 60000.0       # Activation energy for SEI growth [J/mol]
    rho_SEI = 1e5          # SEI film resistivity [Ohm·m]
    A_cell = 0.1           # Electrode area [m²]
    lambda_loss = 0.1      # Lithium-ion loss coefficient [Ah/m]
    alpha_SEI = 0.5        # Transfer coefficient for SEI reaction (αc in paper)
    
    # OCV-SOC Polynomial Coefficients (7th order fit)
    # Fitted from CALCE data, R² = 0.9983, RMSE = 9.82 mV (matches paper exactly)
    # V = a0 + a1*SOC + a2*SOC^2 + ... + a7*SOC^7
    # These coefficients are in ascending order for np.polyval (need to reverse)
    ocv_coeffs = np.array([
        3.086977,      # a0 - 3.09V at SOC=0%
        8.715510,      # a1
        -69.083472,    # a2
        291.996081,    # a3
        -672.555663,   # a4
        860.279577,    # a5
        -572.902078,   # a6
        154.676608     # a7 - gives 4.2V at SOC=100%
    ])
    
    # dOCV/dT coefficient for entropic heat [V/K] (Eq. heat_generation)
    dOCV_dT = -0.0005

# =============================================================================
# Default Smartphone Power Parameters
# Updated to match Section 4.2 component power models
# =============================================================================
class SmartphoneParams:
    """Default parameters for smartphone power consumption modeling.
    
    Component power models from Section 4.2:
    - Display (Eq. display_power)
    - CPU with thermal throttling (Eq. cpu_power_throttled)  
    - Wireless connectivity (Eq. conn_power)
    - GPS (Eq. gps_power)
    - Background tasks (Eq. bg_power)
    - Edge AI (Eq. ai_power_heterogeneous)
    """
    
    # Display (OLED) - Eq. display_power
    P_base_display = 0.1   # Baseline display power (P_base) [W]
    eta_brightness = 1.5   # Brightness power coefficient (η_b) [W]
    eta_red = 0.3          # Red pixel power coefficient (η_R) [W]
    eta_green = 0.4        # Green pixel power coefficient (η_G) [W]
    eta_blue = 0.5         # Blue pixel power coefficient (η_B) [W]
    
    # CPU (DVFS Model) - Eq. cpu_power_throttled
    C_eff = 1e-9           # Effective capacitance (C_eff) [F]
    V_dd_min = 0.6         # Minimum supply voltage [V]
    V_dd_max = 1.0         # Maximum supply voltage [V]
    f_min = 0.5e9          # Minimum CPU frequency [Hz]
    f_max = 3.0e9          # Maximum CPU frequency [Hz]
    P_leak_ref = 0.1       # Reference leakage power at T_ref [W]
    leak_temp_coeff = 0.02 # Temperature coefficient for leakage [1/K]
    
    # Thermal Throttling - Eq. throttling
    T_limit = 318.15       # Temperature limit for throttling (T_limit) [K] (45°C)
    T_thresh = 313.15      # Temperature warning threshold (T_thresh) [K] (40°C)
    V_cutoff = 3.0         # Voltage cutoff (V_cutoff) [V]
    dV_safe = 0.3          # Safe voltage margin (ΔV_safe) [V]
    
    # Wireless Connectivity - Eq. conn_power
    P_WiFi = 0.3           # WiFi power [W]
    P_4G = 1.0             # 4G base power (P_4G) [W]
    P_5G = 1.5             # 5G base power (P_5G) [W]
    alpha_signal_4G = 0.5  # 4G signal strength coefficient (α_s)
    beta_signal_5G = 0.8   # 5G signal strength coefficient (β_s)
    
    # GPS Power - Eq. gps_power
    P_gps_acq = 0.15       # Acquisition phase power [W] (~40mA @ 3.8V)
    P_gps_track = 0.03     # Tracking phase power [W] (~8mA @ 3.8V)
    P_gps_sleep = 0.005    # Sleep/idle power [W]
    kappa_search = 0.1     # Weak signal search overhead coefficient
    
    # Background - Eq. bg_power
    P_idle = 0.15          # System floor idle power (P_idle) [W]
    k_bg = 0.02            # Power per background app (λ_k * P_wake avg) [W]
    
    # AI Workload (Heterogeneous Multi-core) - Eq. ai_power_heterogeneous
    P_static_AI = 0.2      # Base static power for AI (P_static) [W]
    alpha_little = 0.5e-9  # Effective capacitance for Little cores (α_little) [F]
    alpha_medium = 1.0e-9  # Effective capacitance for Medium cores (α_medium) [F]
    alpha_big = 2.0e-9     # Effective capacitance for Big cores (α_big) [F]
    beta_idle_core = 0.3   # Idle core power factor (β)
    
    # Usage Modes Power (average values for HMM)
    # P_base in Table 1 = 4.78W corresponds to Gaming mode base power
    mode_powers = {
        'Idle': 0.2,           # Deep sleep / standby
        'Light': 0.8,          # Light browsing, messaging
        'Busy': 2.0,           # Active app usage
        'Gaming': 4.78,        # High-load gaming (P_base from paper)
        'Video': 1.5,          # Video streaming
        'Navigation': 2.5      # GPS + Screen + Connectivity
    }

# =============================================================================
# Numerical Simulation Parameters
# =============================================================================
class SimulationParams:
    """Parameters for numerical simulation and filtering.
    
    EKF implementation based on Section 5.1 (状态估计算法).
    """
    
    dt = 1.0               # Time step [s]
    SOC_cutoff = 0.05      # SOC cutoff threshold (5%) - from paper
    
    # EKF Parameters (Eq. state_equation, measurement_equation)
    Q_soc = 1e-6           # Process noise variance for SOC (σ_soc²)
    Q_V1 = 1e-4            # Process noise variance for V1
    Q_V2 = 1e-4            # Process noise variance for V2
    R_meas = 1e-4          # Measurement noise variance [V²]
    
    # Adaptive EKF (AEKF) - Eq. adaptive_Q_improved, adaptive_R
    lambda_forget = 0.02   # Forgetting factor for AEKF (λ in [0.01, 0.05])
    
    # Monte Carlo
    n_mc_samples = 200     # Number of Monte Carlo samples for TTE (N=200 in paper)
    
    # ODE Solver (Section 5.3 数值求解算法)
    ode_method = 'RK45'    # ODE integration method
    rtol = 1e-6            # Relative tolerance
    atol = 1e-8            # Absolute tolerance
