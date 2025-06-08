#!/bin/bash

# Black Box Challenge - Your Implementation
# This script takes three parameters and outputs the reimbursement amount
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Python implementation - will pass through PRINT_TRIP_TYPE environment variable if set
python3 calculate_reimbursement.py "$1" "$2" "$3" 