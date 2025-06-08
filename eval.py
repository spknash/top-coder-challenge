#!/usr/bin/env python3

import json
import os
import subprocess
import numpy as np
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def print_header():
    """Print the evaluation header."""
    print("🧾 Black Box Challenge - Reimbursement System Evaluation (Python Version)")
    print("================================================================")
    print()

def check_requirements():
    """Check if the required files exist."""
    # Check if run.sh exists
    if not os.path.isfile("run.sh"):
        print("❌ Error: run.sh not found!")
        print("Please create a run.sh script that takes three parameters:")
        print("  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        print("  and outputs the reimbursement amount")
        sys.exit(1)
    
    # Make run.sh executable
    os.chmod("run.sh", 0o755)
    
    # Check if public cases exist
    if not os.path.isfile("public_cases.json"):
        print("❌ Error: public_cases.json not found!")
        print("Please ensure the public cases file is in the current directory.")
        sys.exit(1)

def load_test_cases():
    """Load test cases from the JSON file."""
    print("📊 Running evaluation against 1,000 test cases...")
    print()
    print("Loading test cases...")
    
    with open("public_cases.json", "r") as f:
        cases = json.load(f)
    
    print(f"Loaded {len(cases)} test cases.")
    return cases

def process_case(case_data):
    """Process a single test case."""
    idx, case = case_data
    
    input_data = case["input"]
    trip_duration = input_data["trip_duration_days"]
    miles_traveled = input_data["miles_traveled"]
    receipts_amount = input_data["total_receipts_amount"]
    expected = case["expected_output"]
    
    try:
        # Run the user's implementation
        result = subprocess.run(
            ["./run.sh", str(trip_duration), str(miles_traveled), str(receipts_amount)],
            capture_output=True,
            text=True,
            timeout=5  # Add timeout to prevent hanging
        )
        
        if result.returncode == 0:
            # Clean output (remove whitespace)
            output = result.stdout.strip()
            
            # Check if output is a valid number
            try:
                actual = float(output)
                
                # Calculate absolute error
                error = abs(actual - expected)
                
                # Return result data
                return {
                    "success": True,
                    "case_num": idx + 1,
                    "expected": expected,
                    "actual": actual,
                    "error": error,
                    "trip_duration": trip_duration,
                    "miles_traveled": miles_traveled,
                    "receipts_amount": receipts_amount
                }
            except ValueError:
                return {
                    "success": False,
                    "case_num": idx + 1,
                    "error_msg": f"Invalid output format: {output}"
                }
        else:
            return {
                "success": False,
                "case_num": idx + 1,
                "error_msg": f"Script failed with error: {result.stderr}"
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "case_num": idx + 1,
            "error_msg": "Script timed out (exceeded 5 seconds)"
        }
    except Exception as e:
        return {
            "success": False,
            "case_num": idx + 1,
            "error_msg": f"Unexpected error: {str(e)}"
        }

def run_evaluation(cases):
    """Run the evaluation against all test cases."""
    start_time = time.time()
    
    successful_results = []
    error_results = []
    
    # Process in parallel with a progress bar
    print("Running test cases (this may take a few minutes)...")
    
    # Determine the number of workers based on CPU count, but limit to 4 to avoid overwhelming the system
    max_workers = min(4, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and create a dictionary mapping futures to their indices
        futures = {executor.submit(process_case, (i, case)): i for i, case in enumerate(cases)}
        
        # Process results as they complete with a progress bar
        with tqdm(total=len(cases), desc="Processing", unit="case") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    successful_results.append(result)
                else:
                    error_results.append(result)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    return successful_results, error_results, elapsed_time

def print_results(successful_results, error_results, elapsed_time):
    """Print the evaluation results."""
    if not successful_results:
        print("❌ No successful test cases!")
        print("")
        print("Your script either:")
        print("  - Failed to run properly")
        print("  - Produced invalid output format")
        print("  - Timed out on all cases")
        print("")
        print("Check the errors below for details.")
        return
    
    # Calculate statistics
    successful_runs = len(successful_results)
    errors = [result["error"] for result in successful_results]
    total_error = sum(errors)
    avg_error = total_error / successful_runs
    max_error = max(errors)
    max_error_case = next(result for result in successful_results if result["error"] == max_error)
    
    # Count matches
    exact_matches = sum(1 for error in errors if error < 0.01)
    close_matches = sum(1 for error in errors if error < 1.0)
    
    # Calculate percentages
    exact_pct = 100 * exact_matches / successful_runs
    close_pct = 100 * close_matches / successful_runs
    
    print("✅ Evaluation Complete!")
    print("")
    print("📈 Results Summary:")
    print(f"  Total test cases: {len(successful_results) + len(error_results)}")
    print(f"  Successful runs: {successful_runs}")
    print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
    print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
    print(f"  Average error: ${avg_error:.2f}")
    print(f"  Maximum error: ${max_error:.2f}")
    print(f"  Evaluation time: {elapsed_time:.2f} seconds")
    print("")
    
    # Calculate score (lower is better)
    total_cases = len(successful_results) + len(error_results)
    score = avg_error * 100 + (total_cases - exact_matches) * 0.1
    print(f"🎯 Your Score: {score:.2f} (lower is better)")
    print("")
    
    # Provide feedback based on exact matches
    if exact_matches == total_cases:
        print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
    elif exact_matches > 950:
        print("🥇 Excellent! You are very close to the perfect solution.")
    elif exact_matches > 800:
        print("🥈 Great work! You have captured most of the system behavior.")
    elif exact_matches > 500:
        print("🥉 Good progress! You understand some key patterns.")
    else:
        print("📚 Keep analyzing the patterns in the interviews and test cases.")
    
    # Display high-error cases
    if exact_matches < total_cases:
        print("")
        print("💡 Tips for improvement:")
        print("  Check these high-error cases:")
        
        # Sort results by error (descending) and show top 5
        high_error_cases = sorted(successful_results, key=lambda x: x["error"], reverse=True)[:5]
        for result in high_error_cases:
            print(f"    Case {result['case_num']}: {result['trip_duration']} days, {result['miles_traveled']} miles, ${result['receipts_amount']} receipts")
            print(f"      Expected: ${result['expected']:.2f}, Got: ${result['actual']:.2f}, Error: ${result['error']:.2f}")
    
    # Show errors if any
    if error_results:
        print("")
        print("⚠️  Errors encountered:")
        for result in error_results[:10]:
            print(f"  Case {result['case_num']}: {result['error_msg']}")
        if len(error_results) > 10:
            print(f"  ... and {len(error_results) - 10} more errors")
    
    print("")
    print("📝 Next steps:")
    print("  1. Fix any script errors shown above")
    print("  2. Ensure your run.sh outputs only a number")
    print("  3. Analyze the patterns in the interviews and public cases")
    print("  4. Test edge cases around trip length and receipt amounts")
    print("  5. Submit your solution via the Google Form when ready!")

def main():
    """Main function to run the evaluation."""
    print_header()
    check_requirements()
    cases = load_test_cases()
    successful_results, error_results, elapsed_time = run_evaluation(cases)
    print_results(successful_results, error_results, elapsed_time)

if __name__ == "__main__":
    main() 