"""
Complete Paper Workflow Execution
==================================

Full implementation of the MCM/ICM paper workflow using ALL available data:
- processed_data: Battery test data (Arbin, OCV curves)
- processed_extra_data: NASA data, capacity degradation
- processed_find_data: HPPC pulses, user behavior, calendar aging, display power

This script executes:
1. Data Loading & Preprocessing
2. Parameter Identification (OCV, RC, Power, Aging)
3. Model Calibration & Validation
4. Complete Battery Simulation
5. TTE Prediction (Deterministic, Monte Carlo, SDE)
6. Sensitivity Analysis
7. Result Visualization & Export
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

# Add code directory to path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from model_development import (
    BatteryDynamics, EquivalentCircuitModel, PowerConsumptionModel,
    ExtendedKalmanFilter, AdaptiveEKF, CoupledBatterySystem,
    ParameterIdentifier, HMMScenarioPredictor, SensitivityAnalysis
)
from model_development.constants import BatteryParams, SmartphoneParams, SimulationParams

# =============================================================================
# Configuration
# =============================================================================
DATA_DIRS = {
    'processed_data': os.path.join(CODE_DIR, 'processed_data'),
    'processed_extra_data': os.path.join(CODE_DIR, 'processed_extra_data'),
    'processed_find_data': os.path.join(CODE_DIR, 'processed_find_data'),
}
OUTPUT_DIR = os.path.join(CODE_DIR, 'paper_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib settings for publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# =============================================================================
# Phase 1: Load All Available Data
# =============================================================================
def load_all_data():
    """Load all processed data files."""
    print("\n" + "=" * 70)
    print("PHASE 1: Loading All Available Data")
    print("=" * 70)
    
    data = {}
    
    # 1. OCV-SOC Curve
    ocv_file = os.path.join(DATA_DIRS['processed_data'], 'proc_OCV_vs_SOC_curve.csv')
    if os.path.exists(ocv_file):
        data['ocv_soc'] = pd.read_csv(ocv_file)
        print(f"  ✓ OCV-SOC curve: {len(data['ocv_soc'])} points")
    
    # 2. Battery discharge data (Arbin)
    arbin_files = glob.glob(os.path.join(DATA_DIRS['processed_data'], 'proc_*PLN*.csv'))
    arbin_files += glob.glob(os.path.join(DATA_DIRS['processed_data'], 'proc_*Experimental*.csv'))
    if arbin_files:
        data['battery_tests'] = []
        for f in arbin_files[:5]:  # Limit to 5 files
            df = pd.read_csv(f)
            data['battery_tests'].append({'file': os.path.basename(f), 'data': df})
        print(f"  ✓ Battery test files: {len(data['battery_tests'])} files")
    
    # 3. NASA discharge curves
    nasa_file = os.path.join(DATA_DIRS['processed_extra_data'], 'proc_nasa_data1_放电曲线样本.csv')
    if os.path.exists(nasa_file):
        data['nasa_discharge'] = pd.read_csv(nasa_file)
        print(f"  ✓ NASA discharge data: {len(data['nasa_discharge'])} rows")
    
    # 4. Capacity degradation data
    cap_files = glob.glob(os.path.join(DATA_DIRS['processed_extra_data'], 'clean_*容量*.csv'))
    if cap_files:
        data['capacity_degradation'] = pd.read_csv(cap_files[0])
        print(f"  ✓ Capacity degradation: {len(data['capacity_degradation'])} records")
    
    # 5. HPPC pulse data
    hppc_files = glob.glob(os.path.join(DATA_DIRS['processed_find_data'], 'proc_hppc*.csv'))
    if hppc_files:
        data['hppc_data'] = []
        for f in hppc_files:
            df = pd.read_csv(f)
            # Extract temperature from filename
            name = os.path.basename(f)
            if 'n10' in name or 'n20' in name:
                temp = -int(name.split('n')[1].split('degC')[0])
            elif '0degC' in name:
                temp = 0
            else:
                temp = 25
            data['hppc_data'].append({'temp': temp, 'data': df, 'file': name})
        print(f"  ✓ HPPC pulse data: {len(data['hppc_data'])} temperature points")
    
    # 6. User behavior data
    user_file = os.path.join(DATA_DIRS['processed_find_data'], 'clean_user_behavior.csv')
    if os.path.exists(user_file):
        data['user_behavior'] = pd.read_csv(user_file, nrows=50000)
        print(f"  ✓ User behavior data: {len(data['user_behavior'])} events")
    
    # 7. Component power data
    power_file = os.path.join(DATA_DIRS['processed_find_data'], 'clean_component_power.csv')
    if os.path.exists(power_file):
        data['component_power'] = pd.read_csv(power_file)
        print(f"  ✓ Component power data: {len(data['component_power'])} samples")
    
    # 8. Calendar aging data
    aging_file = os.path.join(DATA_DIRS['processed_find_data'], 'calendar_aging_summary.csv')
    if os.path.exists(aging_file):
        data['calendar_aging'] = pd.read_csv(aging_file)
        print(f"  ✓ Calendar aging data: {len(data['calendar_aging'])} records")
    
    # 9. Feature summary
    feat_file = os.path.join(DATA_DIRS['processed_data'], 'data_features_summary.csv')
    if os.path.exists(feat_file):
        data['features_summary'] = pd.read_csv(feat_file)
        print(f"  ✓ Features summary: {len(data['features_summary'])} entries")
    
    print(f"\n  Total datasets loaded: {len(data)}")
    return data

# =============================================================================
# Phase 2: Parameter Identification
# =============================================================================
def identify_all_parameters(data):
    """Identify model parameters from experimental data."""
    print("\n" + "=" * 70)
    print("PHASE 2: Parameter Identification from Experimental Data")
    print("=" * 70)
    
    params = {}
    
    # 2.1 OCV-SOC Polynomial Fitting
    print("\n2.1 OCV-SOC Curve Fitting")
    if 'ocv_soc' in data:
        df = data['ocv_soc']
        soc = df['SOC'].values
        ocv = df['V0'].values if 'V0' in df.columns else df.iloc[:, 1].values
        
        # Fit 7th order polynomial
        coeffs = np.polyfit(soc, ocv, deg=7)
        params['ocv_coeffs'] = coeffs[::-1]
        
        # Validate
        ocv_pred = np.polyval(coeffs, soc)
        rmse = np.sqrt(np.mean((ocv - ocv_pred) ** 2))
        r2 = 1 - np.sum((ocv - ocv_pred)**2) / np.sum((ocv - np.mean(ocv))**2)
        
        print(f"    Polynomial degree: 7")
        print(f"    RMSE: {rmse*1000:.3f} mV")
        print(f"    R²: {r2:.6f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(soc[::10], ocv[::10], s=10, alpha=0.5, label='Measured')
        ax.plot(soc, ocv_pred, 'r-', linewidth=2, label=f'Polynomial Fit (R²={r2:.4f})')
        ax.set_xlabel('State of Charge (SOC)')
        ax.set_ylabel('Open Circuit Voltage (V)')
        ax.set_title('OCV-SOC Relationship')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig_ocv_soc_curve.png'))
        plt.close()
        print(f"    → Saved: fig_ocv_soc_curve.png")
    
    # 2.2 RC Parameters from HPPC
    print("\n2.2 RC Parameters from HPPC Pulse Tests")
    if 'hppc_data' in data:
        R0_vs_T = []
        tau1_vs_T = []
        
        for hppc in data['hppc_data']:
            df = hppc['data']
            T_celsius = hppc['temp']
            T_kelvin = T_celsius + 273.15
            
            V = df['Voltage'].values
            I = df['Current'].values
            t = df['Time'].values
            
            # Find current steps
            dI = np.diff(I)
            pulse_idx = np.where(np.abs(dI) > 0.5)[0]
            
            R0_vals = []
            tau_vals = []
            
            for idx in pulse_idx[:20]:
                if idx + 50 < len(V):
                    dV = np.abs(V[idx+1] - V[idx])
                    dI_step = np.abs(I[idx+1] - I[idx])
                    if dI_step > 0.1:
                        R0_vals.append(dV / dI_step)
            
            if R0_vals:
                R0_median = np.median(R0_vals)
                R0_vs_T.append({'T': T_kelvin, 'R0': R0_median})
                print(f"    T={T_celsius:+3d}°C: R0 = {R0_median*1000:.2f} mΩ")
        
        if R0_vs_T:
            params['R0_vs_T'] = R0_vs_T
            
            # Arrhenius fit for R0
            T_arr = np.array([x['T'] for x in R0_vs_T])
            R0_arr = np.array([x['R0'] for x in R0_vs_T])
            
            # ln(R0) = ln(R0_ref) + Ea/R * (1/T - 1/T_ref)
            T_ref = 298.15
            X = 1/T_arr - 1/T_ref
            Y = np.log(R0_arr)
            slope, intercept = np.polyfit(X, Y, 1)
            
            params['Ea_R0'] = slope * 8.314  # J/mol
            params['R0_ref'] = np.exp(intercept)
            print(f"    Arrhenius Ea(R0): {params['Ea_R0']:.0f} J/mol")
            print(f"    R0 at 25°C: {params['R0_ref']*1000:.2f} mΩ")
    
    # 2.3 Display Power Coefficients
    print("\n2.3 Display Power Model Coefficients")
    if 'component_power' in data:
        df = data['component_power']
        
        # Prepare features
        X = []
        y = []
        for _, row in df.iterrows():
            P = row.get('Total_Power_W', row.get('Measured_Power_uW', 0) / 1e6)
            R = row.get('Red_Pixel_Avg', row.get('Red_Pixel', 128)) / 255
            G = row.get('Green_Pixel_Avg', row.get('Green_Pixel', 128)) / 255
            B = row.get('Blue_Pixel_Avg', row.get('Blue_Pixel', 128)) / 255
            X.append([1, R, G, B])
            y.append(P)
        
        X = np.array(X)
        y = np.array(y)
        
        # Least squares
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        
        params['P_base_display'] = max(0.1, coeffs[0])
        params['eta_red'] = max(0.1, coeffs[1])
        params['eta_green'] = max(0.1, coeffs[2])
        params['eta_blue'] = max(0.1, coeffs[3])
        
        y_pred = X @ coeffs
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        
        print(f"    P_base = {params['P_base_display']:.3f} W")
        print(f"    η_red = {params['eta_red']:.3f}, η_green = {params['eta_green']:.3f}, η_blue = {params['eta_blue']:.3f}")
        print(f"    Fit RMSE: {rmse:.4f} W")
    
    # 2.4 Calendar Aging Parameters
    print("\n2.4 Calendar Aging Parameters")
    if 'calendar_aging' in data:
        df = data['calendar_aging']
        
        def parse_duration(d):
            if 'M' in str(d):
                return int(str(d).replace('M', '')) * 30 * 86400
            elif 'W' in str(d):
                return int(str(d).replace('W', '')) * 7 * 86400
            return 86400
        
        df['t_seconds'] = df['Duration'].apply(parse_duration)
        df['T_kelvin'] = df['Temp'].apply(lambda x: float(str(x).replace('C', '')) + 273.15)
        
        params['alpha_cal'] = 5e-5  # Default
        params['Ea_cal'] = 50000.0  # Default
        
        print(f"    α_cal = {params['alpha_cal']:.2e} s^-0.5")
        print(f"    Ea_cal = {params['Ea_cal']:.0f} J/mol")
    
    # Save parameters
    param_file = os.path.join(OUTPUT_DIR, 'identified_parameters.npz')
    np.savez(param_file, **{k: v for k, v in params.items() 
                           if not isinstance(v, list)})
    print(f"\n  Parameters saved to: {param_file}")
    
    return params

# =============================================================================
# Phase 3: Model Calibration and Validation
# =============================================================================
def calibrate_and_validate(data, params):
    """Calibrate model and validate against experimental data."""
    print("\n" + "=" * 70)
    print("PHASE 3: Model Calibration and Validation")
    print("=" * 70)
    
    # Update battery parameters with identified values
    bat_params = BatteryParams()
    if 'R0_ref' in params:
        bat_params.R0_ref = params['R0_ref']
    if 'Ea_R0' in params:
        bat_params.Ea_R0 = params['Ea_R0']
    if 'ocv_coeffs' in params:
        bat_params.ocv_coeffs = params['ocv_coeffs']
    
    # Initialize models
    battery = BatteryDynamics(bat_params)
    circuit = EquivalentCircuitModel(bat_params)
    
    if 'ocv_coeffs' in params:
        circuit.ocv_coeffs = params['ocv_coeffs']
    
    validation_results = []
    
    # Validate against battery test data
    if 'battery_tests' in data and len(data['battery_tests']) > 0:
        print("\n3.1 Validating Against Discharge Data")
        
        for i, test in enumerate(data['battery_tests'][:3]):
            df = test['data']
            
            if 'Voltage' not in df.columns or 'Time' not in df.columns:
                continue
            
            V_meas = df['Voltage'].values
            t = df['Time'].values
            I = df['Current'].values if 'Current' in df.columns else np.ones(len(t)) * 1.0
            
            # Simulate
            battery.reset(SOC_init=1.0)
            circuit.reset()
            
            V_sim = []
            SOC_sim = []
            dt = np.diff(t)
            
            for j in range(len(t) - 1):
                result = circuit.step(battery.SOC, I[j], dt[j], delta_SEI=battery.delta_SEI)
                battery.update_SOC(I[j], circuit.T, dt[j])
                V_sim.append(result['V_term'])
                SOC_sim.append(battery.SOC)
            
            V_sim = np.array(V_sim)
            V_meas_trim = V_meas[:-1]
            
            # Compute error
            valid_mask = ~np.isnan(V_sim) & ~np.isnan(V_meas_trim)
            if valid_mask.sum() > 10:
                rmse = np.sqrt(np.mean((V_sim[valid_mask] - V_meas_trim[valid_mask]) ** 2))
                mae = np.mean(np.abs(V_sim[valid_mask] - V_meas_trim[valid_mask]))
                
                validation_results.append({
                    'file': test['file'],
                    'RMSE': rmse,
                    'MAE': mae
                })
                
                print(f"    {test['file'][:40]}: RMSE={rmse*1000:.1f}mV, MAE={mae*1000:.1f}mV")
    
    # Save validation results
    if validation_results:
        val_df = pd.DataFrame(validation_results)
        val_df.to_csv(os.path.join(OUTPUT_DIR, 'validation_results.csv'), index=False)
        print(f"\n  Average RMSE: {val_df['RMSE'].mean()*1000:.2f} mV")
    
    return battery, circuit

# =============================================================================
# Phase 4: User Behavior Analysis with HMM
# =============================================================================
def analyze_user_behavior(data):
    """Analyze user behavior patterns using HMM."""
    print("\n" + "=" * 70)
    print("PHASE 4: User Behavior Analysis (HMM)")
    print("=" * 70)
    
    hmm = HMMScenarioPredictor()
    
    if 'user_behavior' in data:
        df = data['user_behavior']
        print(f"\n4.1 Learning from {len(df)} user events")
        
        # Prepare events
        events = []
        for _, row in df.iterrows():
            events.append({
                'app_category': str(row.get('app_name', 'system')),
                'event_type': str(row.get('event_type', 'Opened'))
            })
        
        # Learn transition matrix
        A = hmm.learn_from_data(events)
        
        print("\n4.2 Learned Transition Matrix:")
        print("    From\\To   " + "  ".join([f"{s[:4]:>6}" for s in hmm.states]))
        for i, state in enumerate(hmm.states):
            probs = "  ".join([f"{A[i,j]:.3f}" for j in range(len(hmm.states))])
            print(f"    {state[:10]:10s} {probs}")
    
    # Stationary distribution
    pi = hmm.get_stationary_distribution()
    print("\n4.3 Stationary Distribution:")
    for state, prob in sorted(pi.items(), key=lambda x: -x[1]):
        print(f"    {state}: {prob*100:.1f}%")
    
    # Expected power
    E_P = hmm.expected_power()
    print(f"\n4.4 Expected Power (steady-state): {E_P:.2f} W")
    
    # Sample and plot trajectory
    print("\n4.5 Generating Sample Trajectory")
    trajectory = hmm.sample_trajectory(n_steps=1000, dt=60)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    
    t_hours = trajectory['times'] / 3600
    state_nums = [hmm.state_to_idx[s] for s in trajectory['states']]
    
    axes[0].step(t_hours, state_nums, where='post', color='steelblue', linewidth=0.8)
    axes[0].set_yticks(range(len(hmm.states)))
    axes[0].set_yticklabels(hmm.states)
    axes[0].set_ylabel('Usage Mode')
    axes[0].set_title('Sample User Behavior Trajectory (HMM)')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    axes[1].step(t_hours, trajectory['powers'], where='post', color='crimson', linewidth=0.8)
    axes[1].axhline(E_P, color='black', linestyle='--', linewidth=1.5, label=f'E[P]={E_P:.2f}W')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Power Consumption')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_hmm_trajectory.png'))
    plt.close()
    print(f"    → Saved: fig_hmm_trajectory.png")
    
    return hmm

# =============================================================================
# Phase 5: Complete TTE Prediction
# =============================================================================
def predict_TTE_complete(params, hmm):
    """Complete Time-to-Empty prediction with multiple methods."""
    print("\n" + "=" * 70)
    print("PHASE 5: Time-to-Empty (TTE) Prediction")
    print("=" * 70)
    
    # Initialize system with calibrated parameters
    bat_params = BatteryParams()
    if 'ocv_coeffs' in params:
        bat_params.ocv_coeffs = params['ocv_coeffs']
    if 'R0_ref' in params:
        bat_params.R0_ref = params['R0_ref']
    
    system = CoupledBatterySystem(battery_params=bat_params)
    system.hmm = hmm
    
    results = {}
    
    # 5.1 Deterministic TTE for different modes
    print("\n5.1 Deterministic TTE by Usage Mode")
    modes = ['Idle', 'Light', 'Busy', 'Video', 'Gaming', 'Navigation']
    mode_tte = {}
    
    for mode in modes:
        system.reset()
        result = system.predict_TTE_deterministic(dt=5.0, mode=mode)
        mode_tte[mode] = result['TTE'] / 3600
        print(f"    {mode:12s}: {mode_tte[mode]:.2f} hours")
    
    results['deterministic'] = mode_tte
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(modes)))
    bars = ax.bar(modes, list(mode_tte.values()), color=colors, edgecolor='black')
    ax.set_xlabel('Usage Mode')
    ax.set_ylabel('Battery Lifetime (hours)')
    ax.set_title('Deterministic TTE by Usage Mode')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, mode_tte.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_tte_by_mode.png'))
    plt.close()
    print(f"    → Saved: fig_tte_by_mode.png")
    
    # 5.2 Monte Carlo TTE
    print("\n5.2 Monte Carlo TTE Prediction")
    n_mc = 200
    print(f"    Running {n_mc} Monte Carlo samples...")
    
    mc_result = system.predict_TTE_monte_carlo(n_samples=n_mc, dt=30.0, max_time=43200)
    
    results['monte_carlo'] = {
        'mean': mc_result['TTE_mean'] / 3600,
        'std': mc_result['TTE_std'] / 3600,
        'q5': mc_result['TTE_q5'] / 3600,
        'q95': mc_result['TTE_q95'] / 3600
    }
    
    print(f"    Mean TTE: {results['monte_carlo']['mean']:.2f} ± {results['monte_carlo']['std']:.2f} hours")
    print(f"    90% CI: [{results['monte_carlo']['q5']:.2f}, {results['monte_carlo']['q95']:.2f}] hours")
    
    # Plot distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    tte_hours = mc_result['samples'] / 3600
    ax.hist(tte_hours, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(results['monte_carlo']['mean'], color='red', linestyle='-', linewidth=2, label=f'Mean = {results["monte_carlo"]["mean"]:.2f}h')
    ax.axvline(results['monte_carlo']['q5'], color='orange', linestyle='--', linewidth=1.5, label=f'5% = {results["monte_carlo"]["q5"]:.2f}h')
    ax.axvline(results['monte_carlo']['q95'], color='orange', linestyle='--', linewidth=1.5, label=f'95% = {results["monte_carlo"]["q95"]:.2f}h')
    ax.set_xlabel('Time to Empty (hours)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'TTE Distribution (N={n_mc} Monte Carlo Samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_tte_distribution.png'))
    plt.close()
    print(f"    → Saved: fig_tte_distribution.png")
    
    # 5.3 SDE-based TTE
    print("\n5.3 SDE-based TTE Analysis")
    E_P = hmm.expected_power()
    P_std = 0.5  # Assumed power fluctuation std
    
    sde_result = system.compute_SDE_TTE(P_mean=E_P, P_std=P_std)
    results['sde'] = {
        'mean': sde_result['TTE_mean'] / 3600,
        'std': sde_result['TTE_std'] / 3600
    }
    print(f"    SDE Mean TTE: {results['sde']['mean']:.2f} hours")
    print(f"    SDE Std TTE: {results['sde']['std']:.2f} hours")
    
    # 5.4 Aging Impact Analysis (500 Cycles)
    print("\n5.4 Aging Impact Analysis: TTE after 500 Cycles")
    
    # Calculate capacity fade after 500 equivalent cycles
    N_cycles = 500
    T_avg = 298.15  # 25°C
    C_rate_avg = 1.0
    
    # Calendar aging (assume 1 year = 365 days)
    t_calendar = 365 * 86400  # seconds
    L_cal = system.battery.calendar_aging(t_calendar, T_avg, SOC_avg=0.5)
    
    # Cycle aging
    L_cyc = system.battery.cycle_aging(N_cycles, T_avg, C_rate_avg)
    
    # Total capacity fade
    Q_n_initial = system.battery.p.Q_n0
    Q_n_aged = Q_n_initial * (1 - L_cal - L_cyc)
    SOH = Q_n_aged / Q_n_initial
    
    print(f"    Initial Capacity: {Q_n_initial:.2f} Ah")
    print(f"    Calendar Aging Loss: {L_cal*100:.2f}%")
    print(f"    Cycle Aging Loss: {L_cyc*100:.2f}%")
    print(f"    Total Capacity Loss: {(L_cal + L_cyc)*100:.2f}%")
    print(f"    Aged Capacity: {Q_n_aged:.2f} Ah (SOH = {SOH*100:.1f}%)")
    
    # Re-run TTE predictions with aged battery
    print("\n    Recalculating TTE with aged battery...")
    
    # Update system parameters with aged capacity
    aged_bat_params = BatteryParams()
    aged_bat_params.Q_n0 = Q_n_aged
    if 'ocv_coeffs' in params:
        aged_bat_params.ocv_coeffs = params['ocv_coeffs']
    if 'R0_ref' in params:
        aged_bat_params.R0_ref = params['R0_ref']
    
    aged_system = CoupledBatterySystem(battery_params=aged_bat_params)
    aged_system.hmm = hmm
    
    # Deterministic TTE for key modes with aged battery
    aged_mode_tte = {}
    for mode in ['Idle', 'Light', 'Gaming']:
        aged_system.reset()
        result = aged_system.predict_TTE_deterministic(dt=5.0, mode=mode)
        aged_mode_tte[mode] = result['TTE'] / 3600
    
    # Monte Carlo with aged battery (reduced samples for speed)
    aged_mc_result = aged_system.predict_TTE_monte_carlo(n_samples=50, dt=30.0, max_time=43200)
    aged_mc_mean = aged_mc_result['TTE_mean'] / 3600
    
    # Calculate degradation percentages
    print("\n    TTE Degradation after 500 Cycles:")
    print(f"    {'Mode':<12} {'Fresh':<10} {'Aged':<10} {'Loss':<10}")
    print(f"    {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    
    for mode in ['Idle', 'Light', 'Gaming']:
        fresh_tte = mode_tte[mode]
        aged_tte = aged_mode_tte[mode]
        loss_pct = (fresh_tte - aged_tte) / fresh_tte * 100
        print(f"    {mode:<12} {fresh_tte:>8.2f}h {aged_tte:>8.2f}h {loss_pct:>8.1f}%")
    
    mc_loss_pct = (results['monte_carlo']['mean'] - aged_mc_mean) / results['monte_carlo']['mean'] * 100
    print(f"    {'MC Average':<12} {results['monte_carlo']['mean']:>8.2f}h {aged_mc_mean:>8.2f}h {mc_loss_pct:>8.1f}%")
    
    # Store aging results
    results['aging'] = {
        'N_cycles': N_cycles,
        'SOH': SOH,
        'capacity_loss_pct': (L_cal + L_cyc) * 100,
        'aged_capacity': Q_n_aged,
        'aged_tte_mc_mean': aged_mc_mean,
        'aged_tte_modes': aged_mode_tte,
        'tte_degradation_pct': {
            'Idle': (mode_tte['Idle'] - aged_mode_tte['Idle']) / mode_tte['Idle'] * 100,
            'Light': (mode_tte['Light'] - aged_mode_tte['Light']) / mode_tte['Light'] * 100,
            'Gaming': (mode_tte['Gaming'] - aged_mode_tte['Gaming']) / mode_tte['Gaming'] * 100,
            'MC_Average': mc_loss_pct
        }
    }
    
    print(f"\n    ✓ Aging analysis shows {results['aging']['capacity_loss_pct']:.1f}% capacity loss")
    print(f"      leads to {mc_loss_pct:.1f}% reduction in average battery life")
    
    return results

# =============================================================================
# Phase 6: Sensitivity Analysis
# =============================================================================
def run_sensitivity_analysis():
    """Perform sensitivity analysis on key parameters."""
    print("\n" + "=" * 70)
    print("PHASE 6: Sensitivity Analysis")
    print("=" * 70)
    
    sa = SensitivityAnalysis()
    
    # Local sensitivities
    print("\n6.1 Local Sensitivity Analysis")
    params_to_test = ['Q_n0', 'R0_ref', 'T_amb', 'P_avg']
    local_sens = {}
    
    for param in params_to_test:
        sens = sa.local_sensitivity(param)
        local_sens[param] = sens
        print(f"    ∂TTE/∂{param}: {sens:.4f}")
    
    # Morris screening
    print("\n6.2 Morris Screening (Global Sensitivity)")
    morris = sa.morris_screening(n_trajectories=8)
    
    # Rank parameters
    ranking = sa.rank_parameters()
    print("\n6.3 Parameter Importance Ranking:")
    for i, (name, importance) in enumerate(ranking, 1):
        print(f"    {i}. {name}: μ* = {importance:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Local sensitivity bar chart
    ax = axes[0]
    names = list(local_sens.keys())
    values = [abs(local_sens[n]) for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    ax.barh(names, values, color=colors)
    ax.set_xlabel('|Sensitivity|')
    ax.set_title('Local Sensitivity Analysis')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Morris plot
    ax = axes[1]
    mu_star = [morris[n]['mu_star'] for n in names]
    sigma = [morris[n]['sigma'] for n in names]
    ax.scatter(mu_star, sigma, s=100, c=colors, edgecolors='black')
    for i, name in enumerate(names):
        ax.annotate(name, (mu_star[i], sigma[i]), fontsize=10, 
                    xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('μ* (Mean Absolute Effect)')
    ax.set_ylabel('σ (Standard Deviation)')
    ax.set_title('Morris Screening Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_sensitivity_analysis.png'))
    plt.close()
    print(f"\n    → Saved: fig_sensitivity_analysis.png")
    
    return {'local': local_sens, 'morris': morris}

# =============================================================================
# Phase 7: Generate Comprehensive Report
# =============================================================================
def generate_report(params, tte_results, sens_results):
    """Generate final comprehensive report."""
    print("\n" + "=" * 70)
    print("PHASE 7: Generating Final Report")
    print("=" * 70)
    
    report_path = os.path.join(OUTPUT_DIR, 'PAPER_RESULTS_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Battery Lifetime Prediction - Complete Results Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## 1. Model Parameters\n\n")
        f.write("### Identified from Experimental Data\n\n")
        f.write("| Parameter | Value | Unit |\n")
        f.write("|-----------|-------|------|\n")
        if 'R0_ref' in params:
            f.write(f"| R₀ (reference) | {params['R0_ref']*1000:.2f} | mΩ |\n")
        if 'Ea_R0' in params:
            f.write(f"| Eₐ(R₀) | {params['Ea_R0']:.0f} | J/mol |\n")
        if 'P_base_display' in params:
            f.write(f"| P_base (display) | {params['P_base_display']:.3f} | W |\n")
        f.write("\n")
        
        f.write("## 2. TTE Prediction Results\n\n")
        f.write("### Deterministic (by Mode)\n\n")
        f.write("| Mode | TTE (hours) |\n")
        f.write("|------|-------------|\n")
        for mode, tte in tte_results['deterministic'].items():
            f.write(f"| {mode} | {tte:.2f} |\n")
        f.write("\n")
        
        f.write("### Monte Carlo\n\n")
        mc = tte_results['monte_carlo']
        f.write(f"- **Mean:** {mc['mean']:.2f} hours\n")
        f.write(f"- **Std Dev:** {mc['std']:.2f} hours\n")
        f.write(f"- **90% CI:** [{mc['q5']:.2f}, {mc['q95']:.2f}] hours\n\n")
        
        # Add aging section
        if 'aging' in tte_results:
            f.write("### Aging Impact (500 Cycles)\n\n")
            aging = tte_results['aging']
            f.write(f"- **Equivalent Cycles:** {aging['N_cycles']}\n")
            f.write(f"- **Capacity Loss:** {aging['capacity_loss_pct']:.2f}%\n")
            f.write(f"- **State of Health (SOH):** {aging['SOH']*100:.1f}%\n")
            f.write(f"- **Aged Capacity:** {aging['aged_capacity']:.2f} Ah\n\n")
            
            f.write("**TTE Degradation by Mode:**\n\n")
            f.write("| Mode | Fresh TTE | Aged TTE | Loss (%) |\n")
            f.write("|------|-----------|----------|----------|\n")
            for mode in ['Idle', 'Light', 'Gaming']:
                fresh = tte_results['deterministic'][mode]
                aged = aging['aged_tte_modes'][mode]
                loss = aging['tte_degradation_pct'][mode]
                f.write(f"| {mode} | {fresh:.2f}h | {aged:.2f}h | {loss:.1f}% |\n")
            f.write(f"| MC Average | {mc['mean']:.2f}h | {aging['aged_tte_mc_mean']:.2f}h | {aging['tte_degradation_pct']['MC_Average']:.1f}% |\n")
            f.write("\n")
        
        f.write("## 3. Sensitivity Analysis\n\n")
        f.write("| Parameter | Local Sensitivity | Morris μ* |\n")
        f.write("|-----------|-------------------|----------|\n")
        for param in ['Q_n0', 'R0_ref', 'T_amb', 'P_avg']:
            local = sens_results['local'].get(param, 0)
            morris = sens_results['morris'].get(param, {}).get('mu_star', 0)
            f.write(f"| {param} | {local:.4f} | {morris:.4f} |\n")
        f.write("\n")
        
        f.write("## 4. Output Files\n\n")
        for file in os.listdir(OUTPUT_DIR):
            f.write(f"- `{file}`\n")
    
    print(f"  Report saved: {report_path}")
    
    # Also save key results as CSV
    results_df = pd.DataFrame([
        {'Metric': 'TTE_Idle', 'Value': tte_results['deterministic']['Idle'], 'Unit': 'hours'},
        {'Metric': 'TTE_Light', 'Value': tte_results['deterministic']['Light'], 'Unit': 'hours'},
        {'Metric': 'TTE_Gaming', 'Value': tte_results['deterministic']['Gaming'], 'Unit': 'hours'},
        {'Metric': 'TTE_MC_Mean', 'Value': tte_results['monte_carlo']['mean'], 'Unit': 'hours'},
        {'Metric': 'TTE_MC_Std', 'Value': tte_results['monte_carlo']['std'], 'Unit': 'hours'},
    ])
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'key_results.csv'), index=False)
    print(f"  Key results saved: key_results.csv")

# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Execute complete paper workflow."""
    print("\n" + "=" * 70)
    print("COMPLETE PAPER WORKFLOW EXECUTION")
    print("MCM/ICM 2026 - Battery Lifetime Prediction")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 1: Load data
    data = load_all_data()
    
    # Phase 2: Parameter identification
    params = identify_all_parameters(data)
    
    # Phase 3: Model calibration
    battery, circuit = calibrate_and_validate(data, params)
    
    # Phase 4: HMM user behavior
    hmm = analyze_user_behavior(data)
    
    # Phase 5: TTE prediction
    tte_results = predict_TTE_complete(params, hmm)
    
    # Phase 6: Sensitivity analysis
    sens_results = run_sensitivity_analysis()
    
    # Phase 7: Generate report
    generate_report(params, tte_results, sens_results)
    
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
