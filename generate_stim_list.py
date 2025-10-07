"""
A simple script that generates a CSV with a list of all of the stimulus images
used in the experiment.

For each trial, we have:
- target: the reference stimulus
- option_a, option_b: two comparison stimuli

The actual left/right display position is balanced at runtime, not in the CSV.
"""

import csv
from pathlib import Path
from random import shuffle
import re

# Path to the directory containing the stimulus images
stimulus_dir = Path("diskcomp/static/stimuli")

# Path to the CSV file to write the list of stimulus images to
output_file = Path("diskcomp/_private/stim.csv")

# Get a list of all of the stimulus images in the directory
stimulus_images = [f for f in stimulus_dir.iterdir() if f.is_file() and f.suffix == ".png"]

# get the ids of the images
# ID is \d+[a-z]* at the beginning of the filename followed by an underscore or space
stimulus_ids = [m.group(1) for f in stimulus_images if (m := re.match(r"(\d+[a-z]*)(?:_| )", f.name))]
stim_file_map: dict[str, str] = {}
# Write the list of stimulus images to the output file
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "filename"])
    for id, image in zip(stimulus_ids, stimulus_images):
        writer.writerow([id, image.name])
        stim_file_map[id] = image.name


# Generate the list of each possible combination of stimuli
# Each trial consists of three stimuli:
# - target: the reference stimulus at bottom
# - option_a, option_b: two comparison stimuli (left/right position determined at runtime)

output_file = Path("diskcomp/_private/trial_stimuli_unassigned.csv")
n_results = 0
# dict of existing comparisons
# key is target stimulus, value is (option_a, option_b) ...order doesn't matter for the options
comparisons: dict[str, list[tuple[str, ...]]] = {}
sorted_stimuli: list[str] = sorted(stimulus_ids)
trial_combinations: list[tuple[str, str, str]] = []

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["option_a", "option_b", "target"])
    for i, target in enumerate(sorted_stimuli):
        for j, option_a in enumerate(sorted_stimuli):
            if option_a == target:
                continue
            for k, option_b in enumerate(sorted_stimuli):
                if option_b == target or option_b == option_a:
                    continue
                # sort the options to avoid duplicates (a,b,target) == (b,a,target)
                option_pair: tuple[str, ...] = tuple(sorted((option_a, option_b,)))
                if target not in comparisons.keys():
                    comparisons[target] = [option_pair]
                    writer.writerow([option_pair[0], option_pair[1], target])
                    trial_combinations.append((option_pair[0], option_pair[1], target))
                    n_results += 1
                else:
                    if option_pair not in comparisons[target]:
                        comparisons[target].append(option_pair)
                        writer.writerow([option_pair[0], option_pair[1], target])
                        trial_combinations.append((option_pair[0], option_pair[1], target))
                        n_results += 1

print(f"Generated {n_results} unique combinations")

n_participants = 180
n_trials_per_participant = 400
expected_repeats = 3

shuffle(trial_combinations)

# Create a CSV file that assigns trials to participants
# Note: "participant_id" here represents a trial set, not an actual participant
output_file = Path("diskcomp/_private/trial_list.csv")

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["participant_id", "trial_num", "option_a", "option_b", "target", "option_a_file", "option_b_file", "target_file"])
    for p in range(n_participants):
        for t in range(n_trials_per_participant):
            trial_index = (p * n_trials_per_participant + t) % len(trial_combinations)
            option_a, option_b, target = trial_combinations[trial_index]

            writer.writerow([p + 1, t + 1, option_a, option_b, target, stim_file_map[option_a], stim_file_map[option_b], stim_file_map[target]])

print(f"\nCreated trial assignments for {n_participants} trial sets")
print(f"Each trial set has {n_trials_per_participant} trials")

# Verify the assignments
stim_summary: dict[int, int] = {}
with open(output_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    participant_trials: dict[str, int] = {}
    combination_counts: dict[tuple[str, ...], int] = {}
    
    for row in reader:
        participant_id = row[0]
        option_a = row[2]
        option_b = row[3]
        target = row[4]
        
        if participant_id not in participant_trials:
            participant_trials[participant_id] = 1
        else:
            participant_trials[participant_id] += 1
        
        # Normalize combination order for counting
        combination = tuple(sorted([option_a, option_b])) + (target,)
        if combination not in combination_counts:
            combination_counts[combination] = 1
        else:
            combination_counts[combination] += 1
    
    # Check for incorrect trial counts
    for participant_id, n_trials in participant_trials.items():
        if n_trials != n_trials_per_participant:
            print(f"WARNING: Trial set {participant_id} has {n_trials} trials instead of {n_trials_per_participant}")
    
    # Summarize combination frequencies
    for combination, count in combination_counts.items():
        stim_summary[count] = stim_summary.get(count, 0) + 1
        if count < expected_repeats - 1:  # Allow some flexibility
            print(f"WARNING: Combination {combination} appears only {count} times")
    
    print("\n=== Verification Summary ===")
    print(f"Number of trial sets: {len(participant_trials)}")
    print(f"Trials per trial set: {set(participant_trials.values())}")
    print(f"Total unique combinations: {len(combination_counts)}")
    print("\nCombination frequency distribution:")
    for count, n_combinations in sorted(stim_summary.items()):
        print(f"  {n_combinations} combinations appear {count} times")
    print("\nVerification complete!")
