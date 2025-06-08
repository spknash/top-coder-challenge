#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
import glob

def classify_trip_type(days, miles, receipts):
    """
    Classify a trip into one of 8 types based on its characteristics.
    Must match the classification logic in model_trainer.py.
    """
    miles_per_day = miles / days if days > 0 else 0
    receipts_per_day = receipts / days if days > 0 else 0
    
    # Short high-mileage trips
    if days <= 3 and (miles > 500 or miles_per_day > 200):
        return "short_high_mileage"
    
    # Short high-receipt trips
    elif days <= 3 and receipts_per_day > 300:
        return "short_high_receipt"
    
    # Medium-length balanced trips
    elif 4 <= days <= 7 and 50 <= miles_per_day <= 200 and receipts_per_day <= 200:
        return "medium_balanced"
    
    # Medium-length high-receipt trips
    elif 4 <= days <= 7 and receipts_per_day > 250:
        return "medium_high_receipt"
    
    # Long low-activity trips
    elif days >= 8 and miles_per_day < 50:
        return "long_low_activity"
    
    # Long high-efficiency trips
    elif days >= 8 and miles_per_day > 150:
        return "long_high_efficiency"
    
    # Long high-receipt trips
    elif days >= 8 and receipts_per_day > 200:
        return "long_high_receipt"
    
    # Edge case: Very low activity trips
    elif miles_per_day < 10:
        return "very_low_activity"
    
    # Default case
    else:
        return "standard"

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    """
    Calculate reimbursement using a trained machine learning model based on trip type.
    
    If the model file doesn't exist, falls back to a formula-based calculation.
    """
    # Convert inputs to appropriate types
    days = float(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)
    
    # Classify the trip type
    trip_type = classify_trip_type(days, miles, receipts)
    
    # Create feature array for prediction
    features = np.array([[days, miles, receipts]])
    
    # Define model path based on trip type
    models_dir = 'trip_type_models_forest'
    trip_type_model_path = os.path.join(models_dir, trip_type, f"{trip_type}_model.pkl")
    
    # Try to use trip type specific model
    if os.path.exists(trip_type_model_path):
        try:
            with open(trip_type_model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Predict the reimbursement amount
            prediction = model.predict(features)[0]
            
            # Round to 2 decimal places
            return round(prediction, 2), trip_type
        except Exception as e:
            print(f"Error using {trip_type} model: {e}", file=sys.stderr)
            # Fall back to another model or formula-based calculation
    
    # If trip type model not available, try to use any available model
    try:
        model_files = glob.glob(os.path.join(models_dir, '*', '*_model.pkl'))
        if model_files:
            # Use the first available model
            with open(model_files[0], 'rb') as f:
                model = pickle.load(f)
            
            # Predict the reimbursement amount
            prediction = model.predict(features)[0]
            
            # Round to 2 decimal places
            return round(prediction, 2), trip_type
    except Exception as e:
        print(f"Error using fallback model: {e}", file=sys.stderr)
        # Fall back to formula-based calculation
    
    # Fallback calculation based on patterns from interviews
    # Use the trip type to customize the calculation
    
    # Base per diem with adjustments for trip type
    if days == 5:
        per_diem_base = 110  # Special case for 5-day trips
    elif trip_type == "long_low_activity":
        per_diem_base = 90   # Reduced for low activity
    elif trip_type in ["short_high_mileage", "medium_balanced"]:
        per_diem_base = 105  # Bonus for efficient trips
    else:
        per_diem_base = 100  # Standard rate
    
    per_diem_total = days * per_diem_base
    
    # Mileage calculation with different rates based on trip type
    miles_per_day = miles / days if days > 0 else 0
    
    if trip_type == "short_high_mileage":
        # Lower rate for very high mileage trips
        if miles > 800:
            mileage_reimbursement = 800 * 0.50 + (miles - 800) * 0.30
        else:
            mileage_reimbursement = miles * 0.50
    elif trip_type in ["long_low_activity", "very_low_activity"]:
        # Higher rate for trips with little driving
        mileage_reimbursement = miles * 0.60
    else:
        # Standard tiered rates
        if miles <= 100:
            mileage_reimbursement = miles * 0.58
        else:
            mileage_reimbursement = 100 * 0.58 + (miles - 100) * 0.38
    
    # Efficiency bonus/penalty based on trip type
    if trip_type == "medium_balanced" and 150 <= miles_per_day <= 250:
        efficiency_bonus = days * 25  # Bonus for balanced medium trips
    elif trip_type == "short_high_mileage" and miles_per_day > 300:
        efficiency_bonus = 0  # No bonus for unrealistically high mileage
    elif trip_type in ["long_low_activity", "very_low_activity"]:
        efficiency_bonus = -days * 5  # Small penalty for very low activity
    else:
        efficiency_bonus = 0
    
    # Receipt handling based on trip type
    daily_receipt_average = receipts / days if days > 0 else receipts
    
    if trip_type == "short_high_receipt":
        # Cap for very high receipt amounts on short trips
        if daily_receipt_average > 300:
            receipt_adjustment = days * 300 + (receipts - days * 300) * 0.5
        else:
            receipt_adjustment = receipts
    elif trip_type == "medium_high_receipt" or trip_type == "long_high_receipt":
        # Graduated caps for high receipt trips
        if daily_receipt_average > 400:
            receipt_adjustment = days * 400 + (receipts - days * 400) * 0.4
        elif daily_receipt_average > 250:
            receipt_adjustment = days * 250 + (receipts - days * 250) * 0.7
        else:
            receipt_adjustment = receipts
    elif receipts < 10:
        # Penalty for tiny receipts
        receipt_adjustment = -10
    else:
        # Standard receipt handling
        receipt_adjustment = receipts
    
    # Special case bonuses
    if trip_type == "medium_balanced" and 4 <= days <= 6 and 150 <= miles_per_day <= 250:
        special_bonus = 50  # "Sweet spot combo" bonus
    elif trip_type == "short_high_mileage" and days == 1 and miles > 700:
        special_bonus = 30  # One-day road warrior bonus
    else:
        special_bonus = 0
    
    # Final calculation
    reimbursement = per_diem_total + mileage_reimbursement + efficiency_bonus + receipt_adjustment + special_bonus
    
    # Round to 2 decimal places
    return round(reimbursement, 2), trip_type

if __name__ == "__main__":
    # Check if correct number of arguments provided
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    # Get arguments from command line
    trip_duration_days = sys.argv[1]
    miles_traveled = sys.argv[2]
    total_receipts_amount = sys.argv[3]
    
    # Calculate and print reimbursement
    result, trip_type = calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount)
    
    # Check if we should print trip type (for evaluation purposes)
    if os.environ.get('PRINT_TRIP_TYPE') == '1':
        print(f"TRIP_TYPE:{trip_type}")
    
    # Output just the result as required
    print(result) 