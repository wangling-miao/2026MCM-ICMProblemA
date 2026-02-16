"""
Model Development Package for Smartphone Battery Lifetime Prediction
=====================================================================

This package implements the complete continuous-time coupled modeling framework
as described in Section 4 of the MCM/ICM paper.

Modules:
--------
- battery_dynamics: SOC dynamics, capacity fade, SEI aging kinetics
- equivalent_circuit: 2nd-order RC model, Arrhenius compensation, thermal model
- power_consumption: Display, CPU, connectivity, background, AI power models
- state_observer: EKF, Adaptive EKF, PINN-based correction
- coupled_system: Complete system integration and TTE prediction
- parameter_identification: Data-driven parameter calibration
- hmm_scenario: Hidden Markov Model for user behavior prediction
"""

from .battery_dynamics import BatteryDynamics
from .equivalent_circuit import EquivalentCircuitModel
from .power_consumption import PowerConsumptionModel
from .state_observer import ExtendedKalmanFilter, AdaptiveEKF
from .coupled_system import CoupledBatterySystem
from .parameter_identification import ParameterIdentifier
from .hmm_scenario import HMMScenarioPredictor
from .sensitivity_analysis import SensitivityAnalysis
from .pinn_correction import PINNCorrection

__all__ = [
    'BatteryDynamics',
    'EquivalentCircuitModel', 
    'PowerConsumptionModel',
    'ExtendedKalmanFilter',
    'AdaptiveEKF',
    'CoupledBatterySystem',
    'ParameterIdentifier',
    'HMMScenarioPredictor',
    'SensitivityAnalysis',
    'PINNCorrection'
]
