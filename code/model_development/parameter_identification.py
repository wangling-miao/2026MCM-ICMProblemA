"""
Parameter Identification Module
================================

Calibrates model parameters from experimental data:
- OCV-SOC curve fitting from processed data
- RC parameters from HPPC pulse tests
- Power model coefficients from component data
- Aging parameters from calendar aging data
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
import os

class ParameterIdentifier:
    """
    Data-driven parameter identification for all model components.
    """
    
    def __init__(self, data_dirs=None):
        if data_dirs is None:
            base = r"c:\Users\chenp\Documents\比赛\2026美赛\code"
            self.data_dirs = {
                'processed_data': os.path.join(base, 'processed_data'),
                'processed_extra_data': os.path.join(base, 'processed_extra_data'),
                'processed_find_data': os.path.join(base, 'processed_find_data')
            }
        else:
            self.data_dirs = data_dirs
        
        self.params = {}
    
    def fit_ocv_soc_curve(self, ocv_file=None):
        """
        Fit OCV-SOC polynomial from data.
        
        Returns
        -------
        ndarray
            Polynomial coefficients [a0, a1, ..., a7]
        """
        if ocv_file is None:
            ocv_file = os.path.join(self.data_dirs['processed_data'], 
                                    'proc_OCV_vs_SOC_curve.csv')
        
        try:
            df = pd.read_csv(ocv_file)
            soc = df['SOC'].values
            ocv = df['V0'].values if 'V0' in df.columns else df.iloc[:, 1].values
            
            # Fit 7th order polynomial
            coeffs = np.polyfit(soc, ocv, deg=7)
            self.params['ocv_coeffs'] = coeffs[::-1]  # Reverse for polynomial evaluation
            
            # Compute fit quality
            ocv_pred = np.polyval(coeffs, soc)
            rmse = np.sqrt(np.mean((ocv - ocv_pred) ** 2))
            self.params['ocv_rmse'] = rmse
            
            return self.params['ocv_coeffs']
        
        except Exception as e:
            print(f"Error fitting OCV-SOC curve: {e}")
            # Return default coefficients
            return np.array([3.0, 0.8, 0.5, -0.3, 0.2, -0.1, 0.05, -0.02])
    
    def identify_rc_parameters_from_pulse(self, pulse_file=None):
        """
        Identify R0, R1, C1, R2, C2 from HPPC pulse data.
        
        Returns
        -------
        dict
            RC parameters at different temperatures
        """
        if pulse_file is None:
            # Find available HPPC files
            hppc_dir = self.data_dirs['processed_find_data']
            hppc_files = [f for f in os.listdir(hppc_dir) if f.startswith('proc_hppc')]
            if not hppc_files:
                return self._default_rc_params()
            pulse_file = os.path.join(hppc_dir, hppc_files[0])
        
        try:
            df = pd.read_csv(pulse_file)
            
            # Extract voltage and current
            V = df['Voltage'].values
            I = df['Current'].values
            t = df['Time'].values
            
            # Find pulse events (current steps)
            dI = np.diff(I)
            pulse_indices = np.where(np.abs(dI) > 0.5)[0]
            
            R0_values = []
            tau_values = []
            
            for idx in pulse_indices[:10]:  # Analyze first 10 pulses
                if idx + 100 < len(V):
                    # Instantaneous voltage drop = I * R0
                    dV_inst = np.abs(V[idx+1] - V[idx])
                    dI_inst = np.abs(I[idx+1] - I[idx])
                    
                    if dI_inst > 0.1:
                        R0_values.append(dV_inst / dI_inst)
                    
                    # Relaxation time constant from voltage recovery
                    V_relaxation = V[idx+1:idx+100]
                    if len(V_relaxation) > 10:
                        V_norm = (V_relaxation - V_relaxation[-1]) / (V_relaxation[0] - V_relaxation[-1] + 1e-6)
                        # Find where V_norm drops to 1/e
                        idx_tau = np.argmax(V_norm < 0.368)
                        if idx_tau > 0:
                            tau_values.append(t[idx+1+idx_tau] - t[idx+1])
            
            if R0_values:
                self.params['R0_measured'] = np.median(R0_values)
            else:
                self.params['R0_measured'] = 0.05
            
            if tau_values:
                self.params['tau1_measured'] = np.median(tau_values)
            else:
                self.params['tau1_measured'] = 40.0
            
            return self.params
        
        except Exception as e:
            print(f"Error identifying RC parameters: {e}")
            return self._default_rc_params()
    
    def _default_rc_params(self):
        return {
            'R0_measured': 0.05,
            'R1_measured': 0.02,
            'C1_measured': 2000.0,
            'tau1_measured': 40.0,
            'R2_measured': 0.01,
            'C2_measured': 10000.0,
            'tau2_measured': 100.0
        }
    
    def fit_display_power_coefficients(self, power_file=None):
        """
        Fit display power model coefficients from component power data.
        
        Returns
        -------
        dict
            Display power coefficients
        """
        if power_file is None:
            power_file = os.path.join(self.data_dirs['processed_find_data'],
                                      'clean_component_power.csv')
        
        try:
            df = pd.read_csv(power_file)
            
            # Build feature matrix
            features = []
            target = []
            
            for _, row in df.iterrows():
                P = row.get('Total_Power_W', row.get('Measured_Power_uW', 0) / 1e6)
                R = row.get('Red_Pixel_Avg', row.get('Red_Pixel', 0.5))
                G = row.get('Green_Pixel_Avg', row.get('Green_Pixel', 0.5))
                B = row.get('Blue_Pixel_Avg', row.get('Blue_Pixel', 0.5))
                
                # Normalize to [0, 1] if needed
                if R > 1:
                    R, G, B = R/255, G/255, B/255
                
                features.append([1.0, R, G, B])  # [P_base, eta_R, eta_G, eta_B]
                target.append(P)
            
            X = np.array(features)
            y = np.array(target)
            
            # Least squares fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            self.params['P_base_display'] = max(0.1, coeffs[0])
            self.params['eta_red'] = max(0.1, coeffs[1])
            self.params['eta_green'] = max(0.1, coeffs[2])
            self.params['eta_blue'] = max(0.1, coeffs[3])
            
            return self.params
        
        except Exception as e:
            print(f"Error fitting display power: {e}")
            return {'P_base_display': 0.1, 'eta_red': 0.3, 'eta_green': 0.4, 'eta_blue': 0.5}
    
    def fit_calendar_aging(self, aging_file=None):
        """
        Fit calendar aging parameters from storage test data.
        
        Returns
        -------
        dict
            Calendar aging parameters
        """
        if aging_file is None:
            aging_file = os.path.join(self.data_dirs['processed_find_data'],
                                      'calendar_aging_summary.csv')
        
        try:
            df = pd.read_csv(aging_file)
            
            # Parse duration to days
            def parse_duration(d):
                if 'M' in str(d):
                    return int(d.replace('M', '')) * 30
                elif 'W' in str(d):
                    return int(d.replace('W', '')) * 7
                return 0
            
            df['days'] = df['Duration'].apply(parse_duration)
            df['t_seconds'] = df['days'] * 86400
            
            # Parse temperature
            def parse_temp(t):
                return float(str(t).replace('C', ''))
            
            df['T_celsius'] = df['Temp'].apply(parse_temp)
            df['T_kelvin'] = df['T_celsius'] + 273.15
            
            # Fit: L_cal = alpha_cal * sqrt(t) * exp(-Ea/(R*T))
            # Simplified: assume capacity retention = 1 - L_cal
            
            # Use median values for rough estimate
            if 'Capacity' in df.columns and df['Capacity'].notna().any():
                Q_initial = df['Capacity'].max()
                df['L_cal'] = 1 - df['Capacity'] / Q_initial
                
                # Linear regression on transformed data
                valid = df['L_cal'] > 0
                if valid.sum() > 2:
                    x = np.sqrt(df.loc[valid, 't_seconds'].values)
                    y = df.loc[valid, 'L_cal'].values
                    slope = np.polyfit(x, y, 1)[0]
                    self.params['alpha_cal'] = abs(slope)
                else:
                    self.params['alpha_cal'] = 5e-5
            else:
                self.params['alpha_cal'] = 5e-5
            
            return self.params
        
        except Exception as e:
            print(f"Error fitting calendar aging: {e}")
            return {'alpha_cal': 5e-5, 'Ea_cal': 50000.0}
    
    def identify_all_parameters(self):
        """Run all parameter identification procedures."""
        self.fit_ocv_soc_curve()
        self.identify_rc_parameters_from_pulse()
        self.fit_display_power_coefficients()
        self.fit_calendar_aging()
        return self.params
    
    def save_parameters(self, filepath=None):
        """Save identified parameters to file."""
        if filepath is None:
            filepath = os.path.join(self.data_dirs['processed_data'], 
                                    'identified_parameters.npz')
        
        np.savez(filepath, **self.params)
        return filepath
    
    def load_parameters(self, filepath):
        """Load parameters from file."""
        data = np.load(filepath, allow_pickle=True)
        self.params = {k: data[k] for k in data.files}
        return self.params
