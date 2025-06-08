#!/usr/bin/env python3

import json
import random

# Load the full test cases
with open('public_cases.json', 'r') as f:
    full_cases = json.load(f)

# Get the total number of cases
total_cases = len(full_cases)
print(f"Total cases in public_cases.json: {total_cases}")

# Select 100 random cases
num_validation_cases = 100
validation_indices = random.sample(range(total_cases), num_validation_cases)
validation_cases = [full_cases[i] for i in validation_indices]

# Save the validation cases to a new file
with open('validation_cases.json', 'w') as f:
    json.dump(validation_cases, f, indent=2)

print(f"Successfully created validation_cases.json with {num_validation_cases} test cases.") 