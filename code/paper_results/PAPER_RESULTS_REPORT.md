# Battery Lifetime Prediction - Complete Results Report

**Generated:** 2026-02-01 14:19:46

---

## 1. Model Parameters

### Identified from Experimental Data

| Parameter | Value | Unit |
|-----------|-------|------|
| R₀ (reference) | 29.11 | mΩ |
| Eₐ(R₀) | 15162 | J/mol |
| P_base (display) | 4.779 | W |

## 2. TTE Prediction Results

### Deterministic (by Mode)

| Mode | TTE (hours) |
|------|-------------|
| Idle | 24.00 |
| Light | 13.51 |
| Busy | 5.40 |
| Video | 7.21 |
| Gaming | 2.25 |
| Navigation | 4.32 |

### Monte Carlo

- **Mean:** 10.49 hours
- **Std Dev:** 0.41 hours
- **90% CI:** [9.75, 11.15] hours

### Aging Impact (500 Cycles)

- **Equivalent Cycles:** 500
- **Capacity Loss:** 0.22%
- **State of Health (SOH):** 99.8%
- **Aged Capacity:** 2.99 Ah

**TTE Degradation by Mode:**

| Mode | Fresh TTE | Aged TTE | Loss (%) |
|------|-----------|----------|----------|
| Idle | 24.00h | 24.00h | 0.0% |
| Light | 13.51h | 13.52h | -0.0% |
| Gaming | 2.25h | 2.25h | 0.3% |
| MC Average | 10.49h | 10.47h | 0.2% |

## 3. Sensitivity Analysis

| Parameter | Local Sensitivity | Morris μ* |
|-----------|-------------------|----------|
| Q_n0 | 0.8731 | 23081.2500 |
| R0_ref | -0.1028 | 75.0000 |
| T_amb | -0.2567 | 82.5000 |
| P_avg | 0.0000 | 16548.7500 |

## 4. Output Files

- `AGING_ANALYSIS_SUMMARY.md`
- `fig_aging_impact.png`
- `fig_hmm_trajectory.png`
- `fig_ocv_soc_curve.png`
- `fig_sensitivity_analysis.png`
- `fig_tte_by_mode.png`
- `fig_tte_degradation_pct.png`
- `fig_tte_distribution.png`
- `identified_parameters.npz`
- `key_results.csv`
- `PAPER_RESULTS_REPORT.md`
- `param_identification_results.txt`
- `validation_results.csv`
- `workflow_log_20260201_141627.txt`
