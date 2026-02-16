
# Data Summary Report

## 1. Experimental Time-Series Data
**Files:**
- `Experimental_data_aged_cell.csv`
- `Experimental_data_fresh_cell.csv`

**Structure:**
- **Format:** CSV
- **Columns:**
  - `Time`: Time in seconds (float)
  - `Current`: Current in Amperes (float)
  - `Voltage`: Voltage in Volts (float)
  - `Temperature`: Temperature in degrees Celsius (float)

**Description:**
These files contain raw experimental data for battery cells (one aged, one fresh). The data records the electrical and thermal state of the cell over time.

---

## 2. OCV vs SOC Curve
**File:**
- `OCV_vs_SOC_curve.csv`

**Structure:**
- **Format:** CSV
- **Columns:**
  - `SOC`: State of Charge (0.0 to 1.0 range inferred)
  - `V0`: Open Circuit Voltage
  
**Description:**
Defines the relationship between the State of Charge (SOC) and the Open Circuit Voltage (OCV), which is critical for battery modeling and SOC estimation.

---

## 3. PLN Metadata Summary
**File:**
- `PLN_Number_SOC_Temp_StoragePeriod.xlsx`

**Structure:**
- **Format:** Excel (.xlsx)
- **Key Sheet:** `Sheet1`
- **Columns:**
  - `PLN`: Identifier (likely "Plan" number or similar)
  - `SOC`: State of Charge
  - `TEMP`: Temperature
  - `Time`: Storage period or duration (values seem missing in first row)
  - `Discharge Capacity`: Measured capacity
  
**Description:**
This file appears to be a summary or look-up table linking specific "PLN" identifiers to experimental conditions (SOC, Temperature) and outcome metrics (Discharge Capacity).

---

## 4. PLN Test Series Data
**Files:**
- Multiple files, e.g., `10_28_2014_PLN_48_57.xlsx`, `11_7_2014_PLN139_150.xlsx`, `10_21_2014_PLN_1to6.xls`, etc.
- These files cover various date ranges and PLN number ranges.

**Structure:**
- **Format:** Excel (.xlsx / .xls)
- **Organization:** Multi-sheet workbooks.
  - **Sheet `Info`:** Metadata sheet containing the report name, file paths, and other header info. Often contains empty rows at the top.
  - **Data Sheets (e.g., `Channel_1-003`, `Channel_1-007`):** contain the actual time-series test data.
  
**Data Sheet Columns (Typical):**
- `Data_Point`: Index of the data point.
- `Test_Time(s)`: Cumulative test time.
- `Date_Time`: Timestamp of the recording.
- `Step_Time(s)`: Time elapsed within the current step.
- `Step_Index`: Identifier for the test step.
- `Cycle_Index`: Cycle count.
- `Current(A)`: Current.
- `Voltage(V)`: Voltage.
- `Charge_Capacity(Ah)`: Cumulative charge capacity.
- `Discharge_Capacity(Ah)`: Cumulative discharge capacity.
- `Charge_Energy(Wh)`: Charge energy.
- `Discharge_Energy(Wh)`: Discharge energy.
- `dV/dt(V/s)`: Rate of voltage change.
- `Internal_Resistance(Ohm)`: Measured internal resistance.
- Other attributes: `Is_FC_Data`, `AC_Impedance(Ohm)`, `ACI_Phase_Angle(Deg)`.

**Description:**
These are detailed logs from a battery testing system (likely Arbin instruments). Each file aggregates data for multiple test channels or sequences (sheets). They provide high-resolution data on the battery's performance during charge/discharge cycles.
