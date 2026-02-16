"""
Parameter Identification Script
================================
Run this script to identify model parameters from experimental data.
"""

import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_development import ParameterIdentifier
import numpy as np

def main():
    print("="*60)
    print("参数识别：从原始数据重新计算")
    print("="*60)
    
    # Run parameter identification
    pi = ParameterIdentifier()
    params = pi.identify_all_parameters()
    
    print("\n" + "="*60)
    print("参数识别结果")
    print("="*60)
    
    # 1. OCV fitting results
    if 'ocv_coeffs' in params:
        print("\n[1] OCV-SOC 多项式系数 (7阶拟合):")
        coeffs = params['ocv_coeffs']
        for i, c in enumerate(coeffs):
            print(f"    a{i} = {c:.6f}")
        if 'ocv_rmse' in params:
            rmse_mV = params['ocv_rmse'] * 1000
            print(f"    → RMSE = {rmse_mV:.2f} mV")
    
    # 2. RC parameters
    print("\n[2] 电路内阻参数:")
    if 'R0_measured' in params:
        R0_mOhm = params['R0_measured'] * 1000
        print(f"    R0 = {R0_mOhm:.2f} mΩ")
    if 'tau1_measured' in params:
        print(f"    τ1 = {params['tau1_measured']:.2f} s")
    
    # 3. Display power coefficients
    print("\n[3] 显示功耗模型系数:")
    if 'P_base_display' in params:
        print(f"    P_base = {params['P_base_display']:.3f} W")
        print(f"    η_red = {params.get('eta_red', 0.3):.3f}")
        print(f"    η_green = {params.get('eta_green', 0.4):.3f}")
        print(f"    η_blue = {params.get('eta_blue', 0.5):.3f}")
    
    # 4. Calendar aging
    print("\n[4] 日历老化参数:")
    if 'alpha_cal' in params:
        print(f"    α_cal = {params['alpha_cal']:.2e}")
    
    # Save parameters
    save_path = pi.save_parameters()
    print(f"\n✓ 参数已保存至: {save_path}")
    
    print("\n" + "="*60)
    print("参数识别完成!")
    print("="*60)
    
    return params

if __name__ == "__main__":
    main()
