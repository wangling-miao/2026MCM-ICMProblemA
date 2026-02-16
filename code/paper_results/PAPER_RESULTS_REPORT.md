# Battery Lifetime Prediction - Complete Results Report

**Generated:** 2026-02-16 05:28:32

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
| Light | 13.53 |
| Busy | 5.41 |
| Video | 7.21 |
| Gaming | 2.25 |
| Navigation | 4.32 |

### Monte Carlo

- **Mean:** 5.52 hours
- **Std Dev:** 0.33 hours
- **90% CI:** [4.96, 6.08] hours

### Aging Impact (500 Cycles)

- **Equivalent Cycles:** 500
- **Capacity Loss:** 0.22%
- **State of Health (SOH):** 99.8%
- **Aged Capacity:** 2.99 Ah

**TTE Degradation by Mode:**

| Mode | Fresh TTE | Aged TTE | Loss (%) |
|------|-----------|----------|----------|
| Idle | 24.00h | 24.00h | 0.0% |
| Light | 13.53h | 13.52h | 0.0% |
| Gaming | 2.25h | 2.25h | 0.2% |
| MC Average | 5.52h | 5.59h | -1.2% |

## 3. Sensitivity Analysis

| Parameter | Local Sensitivity | Morris μ* |
|-----------|-------------------|----------|
| Q_n0 | 0.9245 | 26651.2500 |
| R0_ref | -0.1026 | 127.5000 |
| T_amb | 0.3088 | 71.2500 |
| P_avg | 0.0514 | 32981.2500 |

## 4. Output Files

- `workflow_log_20260201_141627.txt`
- `framework_diagram.png`
- `param_identification_results.txt`
- `fig_tte_by_mode.png`
- `fig_tte_degradation_pct.png`
- `PAPER_RESULTS_REPORT.md`
- `validation_results.csv`
- `fig_ocv_soc_curve.png`
- `fig_aging_impact.png`
- `fig_tte_distribution.png`
- `key_results.csv`
- `algorithm_flowchart.png`
- `fig_sensitivity_analysis.png`
- `fig_hmm_trajectory.png`
- `identified_parameters.npz`
- `AGING_ANALYSIS_SUMMARY.md`
