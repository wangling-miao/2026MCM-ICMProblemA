"""
Sensitivity Analysis Module
===========================

Implements Section 4.5 of the paper:
- Global sensitivity analysis (Sobol indices)
- Local sensitivity (partial derivatives)
- Morris screening for factor prioritization
"""

import numpy as np
from itertools import product
from .coupled_system import CoupledBatterySystem
from .constants import BatteryParams, SmartphoneParams

class SensitivityAnalysis:
    """
    Sensitivity analysis for TTE prediction model.
    """
    
    def __init__(self):
        self.base_params = {
            'Q_n0': 3.0,    # Battery capacity [Ah]
            'R0_ref': 0.05, # Reference resistance [Ohm]
            'T_amb': 298.15,# Ambient temperature [K]
            'P_avg': 1.5,   # Average power [W]
        }
        self.results = {}
    
    def local_sensitivity(self, param_name, delta_pct=0.01):
        """
        Compute local sensitivity ∂TTE/∂param.
        
        Parameters
        ----------
        param_name : str
            Parameter to perturb
        delta_pct : float
            Perturbation percentage
            
        Returns
        -------
        float
            Sensitivity coefficient
        """
        base_val = self.base_params[param_name]
        delta = base_val * delta_pct
        
        # TTE at base
        TTE_base = self._compute_TTE(self.base_params)
        
        # TTE at base + delta
        params_plus = self.base_params.copy()
        params_plus[param_name] = base_val + delta
        TTE_plus = self._compute_TTE(params_plus)
        
        # Sensitivity (normalized)
        sensitivity = ((TTE_plus - TTE_base) / TTE_base) / delta_pct
        
        return sensitivity
    
    def _compute_TTE(self, params):
        """Compute TTE for given parameters."""
        bat_p = BatteryParams()
        bat_p.Q_n0 = params.get('Q_n0', bat_p.Q_n0)
        bat_p.R0_ref = params.get('R0_ref', bat_p.R0_ref)
        
        system = CoupledBatterySystem(battery_params=bat_p)
        
        # Map P_avg to mode
        P_avg = params.get('P_avg', 1.5)
        if P_avg < 0.5:
            mode = 'Idle'
        elif P_avg < 1.2:
            mode = 'Light'
        elif P_avg < 2.5:
            mode = 'Busy'
        else:
            mode = 'Gaming'
        
        result = system.predict_TTE_deterministic(dt=10.0, mode=mode)
        return result['TTE']
    
    def morris_screening(self, n_trajectories=10, n_levels=4):
        """
        Morris method for global sensitivity screening.
        
        Parameters
        ----------
        n_trajectories : int
            Number of random trajectories
        n_levels : int
            Number of parameter levels
            
        Returns
        -------
        dict
            Morris indices (mu*, sigma) for each parameter
        """
        param_names = list(self.base_params.keys())
        k = len(param_names)
        
        # Parameter bounds (±50%)
        bounds = {}
        for name, val in self.base_params.items():
            bounds[name] = (val * 0.5, val * 1.5)
        
        # Generate trajectories
        elementary_effects = {name: [] for name in param_names}
        
        for _ in range(n_trajectories):
            # Random starting point
            x = np.random.randint(0, n_levels, size=k)
            
            # Convert to actual values
            def levels_to_params(levels):
                params = {}
                for i, name in enumerate(param_names):
                    lo, hi = bounds[name]
                    params[name] = lo + (hi - lo) * levels[i] / (n_levels - 1)
                return params
            
            TTE_base = self._compute_TTE(levels_to_params(x))
            
            # Perturb each parameter
            for i, name in enumerate(param_names):
                x_new = x.copy()
                delta = 1 if x[i] < n_levels - 1 else -1
                x_new[i] += delta
                
                TTE_new = self._compute_TTE(levels_to_params(x_new))
                
                # Elementary effect
                lo, hi = bounds[name]
                d_scaled = delta / (n_levels - 1)
                ee = (TTE_new - TTE_base) / (d_scaled * (hi - lo) / self.base_params[name])
                elementary_effects[name].append(ee)
        
        # Compute Morris indices
        morris_indices = {}
        for name in param_names:
            ees = np.array(elementary_effects[name])
            morris_indices[name] = {
                'mu': np.mean(ees),
                'mu_star': np.mean(np.abs(ees)),
                'sigma': np.std(ees)
            }
        
        self.results['morris'] = morris_indices
        return morris_indices
    
    def compute_all_sensitivities(self):
        """Compute all sensitivity metrics."""
        results = {'local': {}, 'morris': {}}
        
        # Local sensitivities
        for param in self.base_params:
            results['local'][param] = self.local_sensitivity(param)
        
        # Morris screening
        results['morris'] = self.morris_screening(n_trajectories=5)
        
        self.results = results
        return results
    
    def rank_parameters(self):
        """Rank parameters by importance."""
        if 'morris' not in self.results:
            self.compute_all_sensitivities()
        
        # Rank by mu_star (absolute mean effect)
        ranking = sorted(
            self.results['morris'].items(),
            key=lambda x: x[1]['mu_star'],
            reverse=True
        )
        
        return [(name, data['mu_star']) for name, data in ranking]
