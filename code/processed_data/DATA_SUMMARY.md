# Processed Data Summary

## 1. Overview
- **Total Processed Data Files:** 141
- **Validation Plots:** 19
- **Feature Summary:** `all_files_features_summary.csv`

## 2. Processed Data Files (`proc_*.csv`)
**Description:**
Standardized, cleaned, and resampled battery data files (1Hz sampling).

**File Naming Convention:**
- Format: `proc_<OriginalFileName>_<SheetName>.csv`

**Table Structure (Columns):**
- `Data_Point`: Measured value
- `Time`: Test time in seconds (standardized)
- `Step_Time(s)`: Test time in seconds (standardized)
- `Step`: Measured value
- `Cycle_Index`: Measured value
- `Current`: Measured value
- `Voltage`: Measured value
- `Charge_Capacity`: Capacity (Ah)
- `Discharge_Capacity`: Capacity (Ah)
- `Charge_Energy(Wh)`: Measured value
- `Discharge_Energy(Wh)`: Measured value
- `dV/dt(V/s)`: Measured value
- `Internal_Resistance`: Measured value
- `Is_FC_Data`: Measured value
- `AC_Impedance(Ohm)`: Measured value
- `ACI_Phase_Angle(Deg)`: Measured value
- `Voltage_smooth`: Savitzky-Golay filtered Voltage (V)
- `Current_smooth`: Savitzky-Golay filtered Current (A)

**Sample Data (First 5 rows):**
|   Data_Point |   Time |   Step_Time(s) |   Step |   Cycle_Index |   Current |   Voltage |   Charge_Capacity |   Discharge_Capacity |   Charge_Energy(Wh) |   Discharge_Energy(Wh) |   dV/dt(V/s) |   Internal_Resistance |   Is_FC_Data |   AC_Impedance(Ohm) |   ACI_Phase_Angle(Deg) |   Voltage_smooth |   Current_smooth |
|-------------:|-------:|---------------:|-------:|--------------:|----------:|----------:|------------------:|---------------------:|--------------------:|-----------------------:|-------------:|----------------------:|-------------:|--------------------:|-----------------------:|-----------------:|-----------------:|
|          1   |      0 |        10.0086 |      1 |             1 |         0 |   3.84727 |                 0 |                    0 |                   0 |                      0 |  0           |                     0 |            0 |                   0 |                      0 |          3.82618 |       -0.0820721 |
|          1.1 |      1 |        11.0102 |      1 |             1 |         0 |   3.84728 |                 0 |                    0 |                   0 |                      0 |  3.23772e-06 |                     0 |            0 |                   0 |                      0 |          3.82779 |       -0.0721762 |
|          1.2 |      2 |        12.0117 |      1 |             1 |         0 |   3.8473  |                 0 |                    0 |                   0 |                      0 |  6.47545e-06 |                     0 |            0 |                   0 |                      0 |          3.8294  |       -0.0622803 |
|          1.3 |      3 |        13.0133 |      1 |             1 |         0 |   3.84732 |                 0 |                    0 |                   0 |                      0 |  9.71317e-06 |                     0 |            0 |                   0 |                      0 |          3.83101 |       -0.0523843 |
|          1.4 |      4 |        14.0148 |      1 |             1 |         0 |   3.84733 |                 0 |                    0 |                   0 |                      0 |  1.29509e-05 |                     0 |            0 |                   0 |                      0 |          3.83262 |       -0.0424884 |

## 3. Validation Plots (`val_*.png`)
**Description:**
Visualizations comparing the original raw signal vs. the processed (smoothed) signal to verify efficient noise removal without signal distortion.
Includes:
- Voltage/Current vs Time comparison
- Power Spectral Density (PSD) analysis
- Voltage histogram

## 4. Feature Summary (`all_files_features_summary.csv`)
**Description:**
Aggregated features for every processed battery cell/cycle.

**Columns:**
`file`, `sheet`, `output_file`, `discharge_capacity`, `charge_capacity`, `coulombic_efficiency`, `avg_voltage`, `voltage_std`, `voltage_range`, `avg_resistance`, `max_temperature`, `avg_temperature`

**Sample Rows:**
| file                    | sheet         | output_file                                |   discharge_capacity |   charge_capacity |   coulombic_efficiency |   avg_voltage |   voltage_std |   voltage_range |   avg_resistance |   max_temperature |   avg_temperature |
|:------------------------|:--------------|:-------------------------------------------|---------------------:|------------------:|-----------------------:|--------------:|--------------:|----------------:|-----------------:|------------------:|------------------:|
| 10_21_2014_PLN_1to6.xls | Channel_1-007 | proc_10_21_2014_PLN_1to6_Channel_1-007.csv |              1.42163 |           2.06749 |               0.68761  |       3.99361 |      0.251034 |         1.38125 |        0.040779  |               nan |               nan |
| 10_21_2014_PLN_1to6.xls | Channel_1-008 | proc_10_21_2014_PLN_1to6_Channel_1-008.csv |              1.43975 |           2.062   |               0.698227 |       3.98585 |      0.251449 |         1.41431 |        0.0400064 |               nan |               nan |
| 10_21_2014_PLN_1to6.xls | Channel_1-009 | proc_10_21_2014_PLN_1to6_Channel_1-009.csv |              1.56807 |           2.25756 |               0.694588 |       3.99176 |      0.248812 |         1.4137  |        0.0353188 |               nan |               nan |
| 10_21_2014_PLN_1to6.xls | Channel_1-014 | proc_10_21_2014_PLN_1to6_Channel_1-014.csv |              1.55778 |           2.23848 |               0.695908 |       3.98535 |      0.250457 |         1.45019 |        0.0308291 |               nan |               nan |
| 10_21_2014_PLN_1to6.xls | Channel_1-015 | proc_10_21_2014_PLN_1to6_Channel_1-015.csv |              1.57198 |           2.26414 |               0.694295 |       3.99773 |      0.249013 |         1.41515 |        0.033443  |               nan |               nan |
