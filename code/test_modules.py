"""Test script to verify all model development modules."""

import sys
sys.path.insert(0, r'c:\Users\chenp\Documents\比赛\2026美赛\code')

from model_development import (
    BatteryDynamics, EquivalentCircuitModel, PowerConsumptionModel,
    ExtendedKalmanFilter, AdaptiveEKF, CoupledBatterySystem,
    ParameterIdentifier, HMMScenarioPredictor, SensitivityAnalysis,
    PINNCorrection
)

print("=" * 60)
print("MODEL DEVELOPMENT MODULES - VERIFICATION TEST")
print("=" * 60)

# Test 1: Battery Dynamics
print("\n1. Battery Dynamics")
battery = BatteryDynamics()
print(f"   Initial capacity: {battery.p.Q_n0} Ah")
print(f"   SOC after init: {battery.SOC}")
battery.update_SOC(I=1.0, T=298.15, dt=100)
print(f"   SOC after 100s @ 1A discharge: {battery.SOC:.4f}")

# Test 2: Equivalent Circuit
print("\n2. Equivalent Circuit Model")
circuit = EquivalentCircuitModel()
print(f"   R0 at 25C: {circuit.R0(298.15):.4f} Ohm")
print(f"   R0 at 0C: {circuit.R0(273.15):.4f} Ohm")
print(f"   OCV at SOC=0.5: {circuit.OCV(0.5):.4f} V")

# Test 3: Power Consumption
print("\n3. Power Consumption Model")
power = PowerConsumptionModel()
print(f"   Idle power: {power.mode_power('Idle'):.2f} W")
print(f"   Gaming power: {power.mode_power('Gaming'):.2f} W")
breakdown = power.get_power_breakdown(brightness=0.5, f_req=2e9, T=298.15, V_term=3.8)
print(f"   Breakdown: Display={breakdown['P_display']:.2f}W, CPU={breakdown['P_CPU']:.2f}W")

# Test 4: EKF
print("\n4. Extended Kalman Filter")
ekf = AdaptiveEKF(battery, circuit)
result = ekf.step(y_meas=3.7, I=1.0, T=298.15, dt=1.0)
print(f"   Estimated SOC: {result['SOC']:.4f}")
print(f"   SOC std: {result['SOC_std']:.6f}")

# Test 5: HMM
print("\n5. HMM Scenario Predictor")
hmm = HMMScenarioPredictor()
print(f"   States: {hmm.states}")
trajectory = hmm.sample_trajectory(n_steps=10, dt=60)
print(f"   Sample trajectory: {trajectory['states'][:5]}...")
E_P = hmm.expected_power()
print(f"   Expected power: {E_P:.2f} W")

# Test 6: Parameter Identification
print("\n6. Parameter Identification")
identifier = ParameterIdentifier()
ocv_coeffs = identifier.fit_ocv_soc_curve()
if ocv_coeffs is not None:
    print(f"   OCV coeffs fitted: {len(ocv_coeffs)} terms")
else:
    print("   OCV fitting skipped (no data)")

# Test 7: Coupled System
print("\n7. Coupled Battery System")
system = CoupledBatterySystem()
result = system.step(dt=1.0, mode='Light')
print(f"   Step result: SOC={result['SOC']:.4f}, V={result['V_term']:.3f}V")

# Quick TTE estimate
print("\n8. Quick TTE Estimation (5 MC samples)")
mc_result = system.predict_TTE_monte_carlo(n_samples=5, dt=60.0, max_time=14400)
print(f"   Mean TTE: {mc_result['TTE_mean']/3600:.2f} hours")
print(f"   TTE std: {mc_result['TTE_std']/3600:.2f} hours")

# Test 8: Sensitivity Analysis
print("\n9. Sensitivity Analysis")
sa = SensitivityAnalysis()
sens_Q = sa.local_sensitivity('Q_n0')
print(f"   Local sensitivity to Q_n0: {sens_Q:.4f}")

# Test 9: PINN
print("\n10. PINN Correction")
pinn = PINNCorrection(hidden_sizes=[8, 4])
import numpy as np
X_test = np.array([[0.5, 1.0, 298.15], [0.3, 2.0, 308.15]])
V_phys = np.array([3.6, 3.4])
delta_V = pinn.forward(X_test)
print(f"   PINN correction output shape: {delta_V.shape}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
