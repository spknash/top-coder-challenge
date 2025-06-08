#!/bin/bash

# Suhaas' Black Box Challenge Evaluation Script
# Modified version that uses validation_cases.json and outputs detailed error analysis

set -e

# Create results directory if it doesn't exist
mkdir -p results

echo "🧾 Black Box Challenge - Custom Evaluation"
echo "==========================================="
echo

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed!"
    echo "Please install jq to parse JSON files:"
    echo "  macOS: brew install jq"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  CentOS/RHEL: sudo yum install jq"
    exit 1
fi

# Check if bc is available for floating point arithmetic
if ! command -v bc &> /dev/null; then
    echo "❌ Error: bc (basic calculator) is required but not installed!"
    echo "Please install bc for floating point calculations:"
    echo "  macOS: brew install bc"
    echo "  Ubuntu/Debian: sudo apt-get install bc"
    echo "  CentOS/RHEL: sudo yum install bc"
    exit 1
fi

# Check if run.sh exists
if [ ! -f "run.sh" ]; then
    echo "❌ Error: run.sh not found!"
    echo "Please create a run.sh script that takes three parameters:"
    echo "  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    echo "  and outputs the reimbursement amount"
    exit 1
fi

# Make run.sh executable
chmod +x run.sh

# Check if validation cases exist
if [ ! -f "validation_cases.json" ]; then
    echo "❌ Error: validation_cases.json not found!"
    echo "Please ensure the validation cases file is in the current directory."
    exit 1
fi

echo "📊 Running evaluation against validation test cases..."
echo

# Extract all test data upfront in a single jq call for better performance
echo "Extracting test data..."
test_data=$(jq -r '.[] | "\(.input.trip_duration_days):\(.input.miles_traveled):\(.input.total_receipts_amount):\(.expected_output)"' validation_cases.json)

# Convert to arrays for faster access (compatible with bash 3.2+)
test_cases=()
while IFS= read -r line; do
    test_cases+=("$line")
done <<< "$test_data"
num_cases=${#test_cases[@]}

# Initialize counters and arrays
successful_runs=0
exact_matches=0
close_matches=0
total_error="0"
max_error="0"
max_error_case=""
results_array=()
errors_array=()

# Process each test case
for ((i=0; i<num_cases; i++)); do
    if [ $((i % 10)) -eq 0 ]; then
        echo "Progress: $i/$num_cases cases processed..." >&2
    fi
    
    # Extract test case data from pre-loaded array
    IFS=':' read -r trip_duration miles_traveled receipts_amount expected <<< "${test_cases[i]}"
    
    # Run the user's implementation with debug flag to get trip type
    if script_output=$(PRINT_TRIP_TYPE=1 ./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>/dev/null); then
        # Check if output contains trip type
        if [[ $script_output == *"TRIP_TYPE:"* ]]; then
            # Extract trip type and actual output
            trip_type=$(echo "$script_output" | grep "TRIP_TYPE:" | cut -d':' -f2- | tr -d '[:space:]')
            output=$(echo "$script_output" | grep -v "TRIP_TYPE:" | tr -d '[:space:]')
        else
            # No trip type, just the output
            trip_type="unknown"
            output=$(echo "$script_output" | tr -d '[:space:]')
        fi
        
        # Check if output is a valid number
        if [[ $output =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
            actual="$output"
            
            # Calculate absolute error using bc
            error=$(echo "scale=10; if ($actual - $expected < 0) -1 * ($actual - $expected) else ($actual - $expected)" | bc)
            
            # Store result in memory array with additional data for error analysis
            miles_per_day=$(echo "scale=2; $miles_traveled / $trip_duration" | bc)
            receipts_per_day=$(echo "scale=2; $receipts_amount / $trip_duration" | bc)
            results_array+=("$error:$expected:$actual:$trip_duration:$miles_traveled:$receipts_amount:$miles_per_day:$receipts_per_day:$trip_type")
            
            successful_runs=$((successful_runs + 1))
            
            # Check for exact match (within $0.01)
            if (( $(echo "$error < 0.01" | bc -l) )); then
                exact_matches=$((exact_matches + 1))
            fi
            
            # Check for close match (within $1.00)
            if (( $(echo "$error < 1.0" | bc -l) )); then
                close_matches=$((close_matches + 1))
            fi
            
            # Update total error
            total_error=$(echo "scale=10; $total_error + $error" | bc)
            
            # Track maximum error
            if (( $(echo "$error > $max_error" | bc -l) )); then
                max_error="$error"
                max_error_case="Case $((i+1)): $trip_duration days, $miles_traveled miles, \$$receipts_amount receipts"
            fi
            
        else
            errors_array+=("Case $((i+1)): Invalid output format: $output")
        fi
    else
        # Capture stderr for error reporting
        error_msg=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>&1 >/dev/null | tr -d '\n')
        errors_array+=("Case $((i+1)): Script failed with error: $error_msg")
    fi
done

# Calculate and display results
if [ $successful_runs -eq 0 ]; then
    echo "❌ No successful test cases!"
    echo ""
    echo "Your script either:"
    echo "  - Failed to run properly"
    echo "  - Produced invalid output format"
    echo "  - Timed out on all cases"
    echo ""
    echo "Check the errors below for details."
else
    # Calculate average error
    avg_error=$(echo "scale=2; $total_error / $successful_runs" | bc)
    
    # Calculate percentages
    exact_pct=$(echo "scale=1; $exact_matches * 100 / $successful_runs" | bc)
    close_pct=$(echo "scale=1; $close_matches * 100 / $successful_runs" | bc)
    
    echo "✅ Evaluation Complete!"
    echo ""
    echo "📈 Results Summary:"
    echo "  Total test cases: $num_cases"
    echo "  Successful runs: $successful_runs"
    echo "  Exact matches (±\$0.01): $exact_matches (${exact_pct}%)"
    echo "  Close matches (±\$1.00): $close_matches (${close_pct}%)"
    echo "  Average error: \$${avg_error}"
    echo "  Maximum error: \$${max_error}"
    echo ""
    
    # Calculate score (lower is better)
    score=$(echo "scale=2; $avg_error * 100 + ($num_cases - $exact_matches) * 0.1" | bc)
    echo "🎯 Your Score: $score (lower is better)"
    echo ""
    
    # Write summary to file
    {
        echo "BLACK BOX CHALLENGE - EVALUATION SUMMARY"
        echo "========================================"
        echo "Date: $(date)"
        echo ""
        echo "Total test cases: $num_cases"
        echo "Successful runs: $successful_runs"
        echo "Exact matches (±\$0.01): $exact_matches (${exact_pct}%)"
        echo "Close matches (±\$1.00): $close_matches (${close_pct}%)"
        echo "Average error: \$${avg_error}"
        echo "Maximum error: \$${max_error}"
        echo "Score: $score (lower is better)"
    } > results/summary.txt
    
    # Provide feedback based on exact matches
    if [ $exact_matches -eq $num_cases ]; then
        echo "🏆 PERFECT SCORE! You have reverse-engineered the system completely!"
    elif [ $exact_matches -gt 95 ]; then
        echo "🥇 Excellent! You are very close to the perfect solution."
    elif [ $exact_matches -gt 80 ]; then
        echo "🥈 Great work! You have captured most of the system behavior."
    elif [ $exact_matches -gt 50 ]; then
        echo "🥉 Good progress! You understand some key patterns."
    else
        echo "📚 Keep analyzing the patterns in the interviews and test cases."
    fi
    
    echo ""
    echo "💡 High Error Analysis:"
    
    # Sort results by error (descending) in memory
    IFS=$'\n' sorted_high_error=($(printf '%s\n' "${results_array[@]}" | sort -t: -k1 -nr))
    
    # Store the 50 highest error cases in high-error.txt
    {
        echo "HIGH ERROR CASES - TOP 50"
        echo "========================="
        echo "Format: Error | Expected | Actual | Days | Miles | Receipts | Miles/Day | Receipts/Day | Trip Type"
        echo ""
        
        # Output the 50 highest error cases
        for ((j=0; j<50 && j<${#sorted_high_error[@]}; j++)); do
            IFS=: read -r error expected actual trip_duration miles_traveled receipts_amount miles_per_day receipts_per_day trip_type <<< "${sorted_high_error[j]}"
            printf "Error: \$%.2f | Expected: \$%.2f | Actual: \$%.2f | Days: %s | Miles: %s | Receipts: \$%.2f | Miles/Day: %s | Receipts/Day: \$%s | Trip Type: %s\n" \
                "$error" "$expected" "$actual" "$trip_duration" "$miles_traveled" "$receipts_amount" "$miles_per_day" "$receipts_per_day" "$trip_type"
        done
    } > results/high-error.txt
    
    # Show top 5 high error cases in terminal
    echo "Top 5 highest error cases (see results/high-error.txt for full list):"
    for ((j=0; j<5 && j<${#sorted_high_error[@]}; j++)); do
        IFS=: read -r error expected actual trip_duration miles_traveled receipts_amount miles_per_day receipts_per_day trip_type <<< "${sorted_high_error[j]}"
        printf "  Days: %s, Miles: %s, Receipts: \$%.2f (Miles/Day: %s, Receipts/Day: \$%s, Trip Type: %s)\n" \
            "$trip_duration" "$miles_traveled" "$receipts_amount" "$miles_per_day" "$receipts_per_day" "$trip_type"
        printf "    Expected: \$%.2f, Got: \$%.2f, Error: \$%.2f\n" \
            "$expected" "$actual" "$error"
    done
    
    echo ""
    echo "💡 Low Error Analysis:"
    
    # Sort results by error (ascending) in memory
    IFS=$'\n' sorted_low_error=($(printf '%s\n' "${results_array[@]}" | sort -t: -k1 -n))
    
    # Store the 50 lowest error cases in low-error.txt
    {
        echo "LOW ERROR CASES - TOP 50"
        echo "========================"
        echo "Format: Error | Expected | Actual | Days | Miles | Receipts | Miles/Day | Receipts/Day | Trip Type"
        echo ""
        
        # Output the 50 lowest error cases
        for ((j=0; j<50 && j<${#sorted_low_error[@]}; j++)); do
            IFS=: read -r error expected actual trip_duration miles_traveled receipts_amount miles_per_day receipts_per_day trip_type <<< "${sorted_low_error[j]}"
            printf "Error: \$%.2f | Expected: \$%.2f | Actual: \$%.2f | Days: %s | Miles: %s | Receipts: \$%.2f | Miles/Day: %s | Receipts/Day: \$%s | Trip Type: %s\n" \
                "$error" "$expected" "$actual" "$trip_duration" "$miles_traveled" "$receipts_amount" "$miles_per_day" "$receipts_per_day" "$trip_type"
        done
    } > results/low-error.txt
    
    # Show top 5 low error cases in terminal
    echo "Top 5 lowest error cases (see results/low-error.txt for full list):"
    for ((j=0; j<5 && j<${#sorted_low_error[@]}; j++)); do
        IFS=: read -r error expected actual trip_duration miles_traveled receipts_amount miles_per_day receipts_per_day trip_type <<< "${sorted_low_error[j]}"
        printf "  Days: %s, Miles: %s, Receipts: \$%.2f (Miles/Day: %s, Receipts/Day: \$%s, Trip Type: %s)\n" \
            "$trip_duration" "$miles_traveled" "$receipts_amount" "$miles_per_day" "$receipts_per_day" "$trip_type"
        printf "    Expected: \$%.2f, Got: \$%.2f, Error: \$%.2f\n" \
            "$expected" "$actual" "$error"
    done
    
    # Generate trip type error analysis
    echo ""
    echo "💡 Trip Type Analysis:"
    
    # Count errors by trip type
    declare -A trip_type_counts
    declare -A trip_type_errors
    
    for result in "${results_array[@]}"; do
        IFS=: read -r error expected actual trip_duration miles_traveled receipts_amount miles_per_day receipts_per_day trip_type <<< "$result"
        
        # Count occurrences
        if [[ -z "${trip_type_counts[$trip_type]}" ]]; then
            trip_type_counts[$trip_type]=1
            trip_type_errors[$trip_type]=$error
        else
            trip_type_counts[$trip_type]=$((trip_type_counts[$trip_type] + 1))
            trip_type_errors[$trip_type]=$(echo "scale=10; ${trip_type_errors[$trip_type]} + $error" | bc)
        fi
    done
    
    # Output trip type error analysis to terminal and file
    {
        echo "TRIP TYPE ERROR ANALYSIS"
        echo "======================="
        echo ""
        echo "Format: Trip Type | Count | Average Error | % of Total Cases"
        echo ""
        
        for trip_type in "${!trip_type_counts[@]}"; do
            count=${trip_type_counts[$trip_type]}
            avg_type_error=$(echo "scale=2; ${trip_type_errors[$trip_type]} / $count" | bc)
            percentage=$(echo "scale=1; $count * 100 / $successful_runs" | bc)
            
            printf "%-20s | %5d | \$%-12s | %s%%\n" "$trip_type" "$count" "$avg_type_error" "$percentage"
            echo "$trip_type:$count:$avg_type_error:$percentage" >> temp_trip_types.txt
        done
    } > results/trip_type_analysis.txt
    
    # Sort by count and show top trip types
    echo "Top trip types by count:"
    sort -t: -k2 -nr temp_trip_types.txt | head -5 | while IFS=: read -r trip_type count avg_error percentage; do
        printf "  %-20s: %d cases (%.1f%%), Avg Error: \$%s\n" "$trip_type" "$count" "$percentage" "$avg_error"
    done
    rm -f temp_trip_types.txt
    
    echo ""
    echo "Trip types with highest average error:"
    sort -t: -k3 -nr temp_trip_types.txt 2>/dev/null | head -5 | while IFS=: read -r trip_type count avg_error percentage; do
        printf "  %-20s: \$%s avg error (%d cases)\n" "$trip_type" "$avg_error" "$count"
    done 2>/dev/null
fi

# Show errors if any
if [ ${#errors_array[@]} -gt 0 ]; then
    echo
    echo "⚠️  Errors encountered:"
    for ((j=0; j<${#errors_array[@]} && j<10; j++)); do
        echo "  ${errors_array[j]}"
    done
    if [ ${#errors_array[@]} -gt 10 ]; then
        echo "  ... and $((${#errors_array[@]} - 10)) more errors"
    fi
fi

echo
echo "📝 Next steps:"
echo "  1. Check results/summary.txt for overall performance"
echo "  2. Review results/high-error.txt to identify patterns in high-error cases"
echo "  3. Study results/low-error.txt to understand what your model gets right"
echo "  4. Analyze results/trip_type_analysis.txt for per-trip-type performance"
echo "  5. Adjust your algorithm based on these insights"
echo "  6. Re-run this evaluation to measure improvement"

# Print path to results
echo 
echo "📊 Results saved to:"
echo "  - results/summary.txt"
echo "  - results/high-error.txt"
echo "  - results/low-error.txt"
echo "  - results/trip_type_analysis.txt" 