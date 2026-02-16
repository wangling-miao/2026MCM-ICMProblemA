"""
Main Simulation Script
======================

Entry point for running the complete battery lifetime prediction model.
Demonstrates:
1. Parameter identification from data
2. System simulation under various scenarios
3. TTE prediction with uncertainty quantification
4. Visualization of results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import model modules
from model_development import (
    BatteryDynamics, EquivalentCircuitModel, PowerConsumptionModel,
    ExtendedKalmanFilter, AdaptiveEKF, CoupledBatterySystem,
    ParameterIdentifier, HMMScenarioPredictor
)
from model_development.constants import BatteryParams, SmartphoneParams, SimulationParams

# Set up output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'simulation_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def identify_parameters():
    """Run parameter identification from experimental data."""
    print("=" * 60)
    print("PHASE 1: Parameter Identification")
    print("=" * 60)
    
    identifier = ParameterIdentifier()
    params = identifier.identify_all_parameters()
    
    print("\nIdentified Parameters:")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value[:3]}... (shape: {value.shape})")
        else:
            print(f"  {key}: {value}")
    
    # Save parameters
    param_file = identifier.save_parameters()
    print(f"\nParameters saved to: {param_file}")
    
    return params

def run_single_discharge_simulation(mode='Light', dt=1.0):
    """
    Simulate a single discharge cycle under constant load.
    
    Parameters
    ----------
    mode : str
        Usage mode
    dt : float
        Time step [s]
    """
    print("\n" + "=" * 60)
    print(f"PHASE 2: Single Discharge Simulation (Mode: {mode})")
    print("=" * 60)
    
    # Initialize coupled system
    system = CoupledBatterySystem()
    
    # Run deterministic TTE prediction
    result = system.predict_TTE_deterministic(dt=dt, mode=mode)
    
    TTE = result['TTE']
    hist = result['history']
    
    print(f"\nResults for {mode} mode:")
    print(f"  Time to Empty: {TTE/3600:.2f} hours ({TTE:.0f} seconds)")
    print(f"  Final SOC: {hist['SOC'][-1]:.4f}")
    print(f"  Average Power: {np.mean(hist['P_load']):.2f} W")
    print(f"  Max Temperature: {np.max(hist['T']) - 273.15:.1f} °C")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    t_hours = hist['time'] / 3600
    
    # SOC
    axes[0, 0].plot(t_hours, hist['SOC'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('SOC')
    axes[0, 0].set_title('State of Charge')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1.05])
    
    # Voltage
    axes[0, 1].plot(t_hours, hist['V_term'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].set_title('Terminal Voltage')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature
    T_celsius = np.array(hist['T']) - 273.15
    axes[1, 0].plot(t_hours, T_celsius, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_title('Battery Temperature')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power
    axes[1, 1].plot(t_hours, hist['P_load'], 'm-', linewidth=1, alpha=0.7)
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].set_title('Power Consumption')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, f'discharge_{mode.lower()}.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    plt.close()
    
    return result

def run_monte_carlo_TTE(n_samples=200):
    """
    Monte Carlo TTE prediction with uncertainty quantification.
    
    Parameters
    ----------
    n_samples : int
        Number of MC samples
    """
    print("\n" + "=" * 60)
    print(f"PHASE 3: Monte Carlo TTE Prediction (N={n_samples})")
    print("=" * 60)
    
    system = CoupledBatterySystem()
    mc_result = system.predict_TTE_monte_carlo(n_samples=n_samples, dt=30.0)
    
    print("\nMonte Carlo Results:")
    print(f"  Mean TTE: {mc_result['TTE_mean']/3600:.2f} hours")
    print(f"  Std TTE: {mc_result['TTE_std']/3600:.2f} hours")
    print(f"  Median TTE: {mc_result['TTE_median']/3600:.2f} hours")
    print(f"  5% percentile: {mc_result['TTE_q5']/3600:.2f} hours")
    print(f"  95% percentile: {mc_result['TTE_q95']/3600:.2f} hours")
    
    # Plot TTE distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tte_hours = mc_result['samples'] / 3600
    ax.hist(tte_hours, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(mc_result['TTE_mean']/3600, color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(mc_result['TTE_q5']/3600, color='orange', linestyle=':', linewidth=2, label='5%')
    ax.axvline(mc_result['TTE_q95']/3600, color='orange', linestyle=':', linewidth=2, label='95%')
    
    ax.set_xlabel('Time to Empty (hours)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title('TTE Distribution (Monte Carlo Simulation)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig_path = os.path.join(OUTPUT_DIR, 'tte_distribution.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    plt.close()
    
    return mc_result

def compare_usage_modes():
    """Compare TTE across different usage modes."""
    print("\n" + "=" * 60)
    print("PHASE 4: Usage Mode Comparison")
    print("=" * 60)
    
    modes = ['Idle', 'Light', 'Busy', 'Video', 'Gaming', 'Navigation']
    results = {}
    
    for mode in modes:
        system = CoupledBatterySystem()
        result = system.predict_TTE_deterministic(dt=5.0, mode=mode)
        results[mode] = result['TTE'] / 3600  # Convert to hours
        print(f"  {mode:12s}: {results[mode]:.2f} hours")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(modes)))
    bars = ax.bar(modes, list(results.values()), color=colors, edgecolor='black')
    
    ax.set_xlabel('Usage Mode', fontsize=12)
    ax.set_ylabel('Time to Empty (hours)', fontsize=12)
    ax.set_title('Battery Lifetime by Usage Mode', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{val:.1f}h', ha='center', va='bottom', fontsize=10)
    
    fig_path = os.path.join(OUTPUT_DIR, 'mode_comparison.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    plt.close()
    
    return results

def run_hmm_scenario_analysis():
    """Analyze user behavior patterns from HMM."""
    print("\n" + "=" * 60)
    print("PHASE 5: HMM User Behavior Analysis")
    print("=" * 60)
    
    hmm = HMMScenarioPredictor()
    
    # Load and learn from user behavior data if available
    data_file = os.path.join(os.path.dirname(__file__), 
                             'processed_find_data', 'clean_user_behavior.csv')
    
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file, nrows=10000)  # Limit for speed
            events = df.to_dict('records')
            hmm.learn_from_data(events)
            print("\nLearned HMM from user behavior data")
        except Exception as e:
            print(f"\nUsing default HMM (could not load data: {e})")
    
    # Get stationary distribution
    pi = hmm.get_stationary_distribution()
    print("\nStationary Distribution:")
    for state, prob in sorted(pi.items(), key=lambda x: -x[1]):
        print(f"  {state:12s}: {prob*100:.1f}%")
    
    # Expected power
    E_P = hmm.expected_power()
    print(f"\nExpected Power (steady-state): {E_P:.2f} W")
    
    # Sample trajectory
    trajectory = hmm.sample_trajectory(n_steps=500, dt=60.0)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    t_min = trajectory['times'] / 60
    
    # State sequence
    state_nums = [hmm.state_to_idx[s] for s in trajectory['states']]
    axes[0].step(t_min, state_nums, where='post', color='steelblue', linewidth=1)
    axes[0].set_yticks(range(len(hmm.states)))
    axes[0].set_yticklabels(hmm.states)
    axes[0].set_xlabel('Time (minutes)')
    axes[0].set_ylabel('Usage Mode')
    axes[0].set_title('HMM State Trajectory')
    axes[0].grid(True, alpha=0.3)
    
    # Power trajectory
    axes[1].step(t_min, trajectory['powers'], where='post', color='crimson', linewidth=1)
    axes[1].axhline(E_P, color='black', linestyle='--', label=f'E[P] = {E_P:.2f} W')
    axes[1].set_xlabel('Time (minutes)')
    axes[1].set_ylabel('Power (W)')
    axes[1].set_title('Power Consumption Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, 'hmm_trajectory.png')
    plt.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")
    plt.close()
    
    return hmm

def generate_summary_report():
    """Generate a summary report of all results."""
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    report_path = os.path.join(OUTPUT_DIR, 'simulation_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Battery Lifetime Prediction - Simulation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Configuration\n\n")
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Battery Capacity | {BatteryParams.Q_n0} Ah |\n")
        f.write(f"| Cutoff Voltage | {BatteryParams.V_cutoff} V |\n")
        f.write(f"| SOC Cutoff | 5% |\n")
        f.write(f"| Reference Temperature | 25°C |\n\n")
        
        f.write("## Results Summary\n\n")
        f.write("See generated figures in this directory for detailed results.\n\n")
        
        f.write("## Files Generated\n\n")
        for file in os.listdir(OUTPUT_DIR):
            f.write(f"- {file}\n")
    
    print(f"Report saved: {report_path}")

def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("BATTERY LIFETIME PREDICTION - MODEL DEVELOPMENT")
    print("MCM/ICM 2026 Implementation")
    print("=" * 60 + "\n")
    
    # Phase 1: Parameter Identification
    params = identify_parameters()
    
    # Phase 2: Single Discharge Simulations
    for mode in ['Light', 'Gaming']:
        run_single_discharge_simulation(mode=mode, dt=1.0)
    
    # Phase 3: Monte Carlo TTE
    mc_result = run_monte_carlo_TTE(n_samples=100)
    
    # Phase 4: Mode Comparison
    mode_results = compare_usage_modes()
    
    # Phase 5: HMM Analysis
    hmm = run_hmm_scenario_analysis()
    
    # Generate Report
    generate_summary_report()
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
