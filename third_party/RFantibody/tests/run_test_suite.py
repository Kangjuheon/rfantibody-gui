#!/usr/bin/env python3

'''
    This script will run each of the test commands in the current directory. The commands are all
    in deterministic mode, so the output should be the same every time. This script will compare the 
    output of each command to the reference output in the reference_output directory. Any differences
    will be noted and printed to the console.

'''

import os
import sys
import subprocess
import re
import torch

if not os.path.exists('example_outputs'):

    # Check if cuda is available
    if not torch.cuda.is_available():
        print('No GPU found, the test suite much be run on an A4000 GPU')
        sys.exit(1)

    # Make sure that we are on an A4000 gpu
    gpu_info = torch.cuda.get_device_properties(0)
    if 'A4000' not in gpu_info.name:
        print('This script must be run on an A4000 GPU')
        sys.exit(1)

    # Get .sh files in current directory
    for command in os.listdir('.'):

        # Skip non .sh files
        if not command.endswith('.sh'):
            continue

        # Run command and wait for it to finish
        print('Running command: ' + command)
        subprocess.run(['bash', command]) 
else:
    print('example_outputs directory already exists. Skipping command execution.')

### Compare Output to Reference Output
###########################################################

# Iterate over all files in reference_output directory
# Find the corresponding output file in the example_output directory and 
# compare the two files line by line

# Get reference output files
ref_output_files = os.listdir('reference_outputs')

ncorrect = 0
ntotal = 0

for ref in ref_output_files:

    # Skip .trb files
    if ref.endswith('.trb'):
        continue

    reffile = f'reference_outputs/{ref}'

    # Get the corresponding output file
    output_file = 'example_outputs/' + ref

    # Compare the two files line by line
    correct = True

    print('Comparing ' + reffile + ' to ' + output_file)

    with open(reffile, 'r') as ref_file, open(output_file, 'r') as output:
        ref_lines = ref_file.readlines()
        output_lines = output.readlines()

        # Compare each line
        for i in range(len(ref_lines)):
            ref_line = ref_lines[i]
            output_line = output_lines[i]

            # Remove whitespace and compare
            ref_line = ref_line.strip()
            output_line = output_line.strip()

            # Compare
            if ref_line != output_line:
                correct = False
                print('Difference in file: ' + ref_base)
                print('Line ' + str(i+1) + ' in reference output:')
                print(ref_line)
                print('Line ' + str(i+1) + ' in example output:')
                print(output_line)
                print()
    
    if correct:
        ncorrect += 1
    ntotal += 1

print('Correct: ' + str(ncorrect) + '/' + str(ntotal))
