"""
Complete Workflow Runner with Logging
======================================
Runs the complete paper workflow and saves all output to a log file.
"""

import os
import sys
import io
from datetime import datetime

# Redirect output to both console and file
class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

def main():
    # Setup output directory
    CODE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(CODE_DIR, 'paper_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(OUTPUT_DIR, f'workflow_log_{timestamp}.txt')
    
    # Setup tee output
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("="*70)
        print("COMPLETE PAPER WORKFLOW EXECUTION")
        print("MCM/ICM 2026 - Battery Lifetime Prediction")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log File: {log_file}")
        print("="*70)
        
        # Import and run the workflow
        sys.path.insert(0, CODE_DIR)
        
        # Import workflow functions
        from run_complete_workflow import (
            load_all_data, identify_all_parameters, calibrate_and_validate,
            analyze_user_behavior, predict_TTE_complete, run_sensitivity_analysis,
            generate_report, OUTPUT_DIR as RESULT_DIR
        )
        
        # Execute phases
        print("\n" + "="*70)
        print("Starting workflow execution...")
        print("="*70)
        
        # Phase 1: Load data
        data = load_all_data()
        
        # Phase 2: Parameter identification
        params = identify_all_parameters(data)
        
        # Phase 3: Model calibration
        battery, circuit = calibrate_and_validate(data, params)
        
        # Phase 4: HMM user behavior
        hmm = analyze_user_behavior(data)
        
        # Phase 5: TTE prediction
        tte_results = predict_TTE_complete(params, hmm)
        
        # Phase 6: Sensitivity analysis
        sens_results = run_sensitivity_analysis()
        
        # Phase 7: Generate report
        generate_report(params, tte_results, sens_results)
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETE!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {RESULT_DIR}")
        print(f"Log saved to: {log_file}")
        print("="*70)
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Restore stdout and close log
        sys.stdout = tee.terminal
        tee.close()
        print(f"\nWorkflow completed. Log saved to: {log_file}")

if __name__ == "__main__":
    main()
