"""
A simple script that generates a CSV with a list of all of the stimulus images
used in the experiment.
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
# followed by the rest of the filename
stimulus_ids = [m.group(1) for f in stimulus_images if (m := re.match(r"(\d+[a-z]*)(?:_| )", f.name))]

# Write the list of stimulus images to the output file
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "filename"])
    for id, image in zip(stimulus_ids, stimulus_images):
        writer.writerow([id, image.name])


# Generate the list of each possible combination of stimuli
# Each trial consists of three stimuli, one on the left and one on the right
# then one in the center-bottom, the participant has to choose which of the two
# side stimuli is closer to the center-bottom stimulus
# The left and right stimuli must be different to each other and different
# from the center-bottom stimulus, order does not matter

output_file = Path("diskcomp/_private/trial_stimuli_unassigned.csv")
n_results = 0
# dict of existing comparisons
# key is center stimulus, value is (left, right) ...order doesn't matter for the sides
comparisons: dict[str, list[tuple[str, ...]]] = {}
sorted_stimuli: list[str] = sorted(stimulus_ids)
trial_combinations: list[tuple[str, str, str]] = []
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["left", "right", "target"])
    for i, center in enumerate(sorted_stimuli):
        for j, left in enumerate(sorted_stimuli):
            if left == center:
                continue
            for k, right in enumerate(sorted_stimuli):
                if right == center or right == left:
                    continue
                # sort the left and right to avoid duplicates
                left_right: tuple[str, ...] = tuple(sorted((left, right,)))
                if center not in comparisons.keys():
                    comparisons[center] = [left_right]
                    writer.writerow([left, right, center])
                    trial_combinations.append((left, right, center))
                    n_results += 1
                else:
                    if left_right not in comparisons[center]:
                        comparisons[center].append(left_right)
                        writer.writerow([left, right, center])
                        trial_combinations.append((left, right, center))
                        n_results += 1

print(f"Generated {n_results} combinations")

n_participants = 180
n_trials_per_participant = 400
expected_repeats = 3

shuffle(trial_combinations)
# Create a CSV file that assigns trials to participants
output_file = Path("diskcomp/_private/trial_stimuli_assigned.csv")
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["participant_id", "trial_num", "left", "right", "target"])
    for p in range(n_participants):
        for t in range(n_trials_per_participant):
            trial_index = (p * n_trials_per_participant + t) % len(trial_combinations)
            left, right, target = trial_combinations[trial_index]
            writer.writerow([p + 1, t + 1, left, right, target])

# read file back in to verify the number of trials per participant
# number of times each combination appears across all participants
stim_summary: dict[int, int] = {}
with open(output_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    participant_trials: dict[str, int] = {}
    combination_counts: dict[tuple[str, str, str], int] = {}
    for row in reader:
        participant_id = row[0]
        left = row[2]
        right = row[3]
        target = row[4]
        if participant_id not in participant_trials:
            participant_trials[participant_id] = 1
        else:
            participant_trials[participant_id] += 1
        combination = (left, right, target)
        if combination not in combination_counts:
            combination_counts[combination] = 1
        else:
            combination_counts[combination] += 1
    for participant_id, n_trials in participant_trials.items():
        if n_trials != n_trials_per_participant:
            print(f"WARNING: Participant {participant_id} has {n_trials} trials instead of {n_trials_per_participant}")
    
    for combination, count in combination_counts.items():
        stim_summary[count] = stim_summary.get(count, 0) + 1
        if count < expected_repeats:
            print(f"WARNING: Combination {combination} appears {count} times instead of {expected_repeats}")
    print("Number of participants:", len(participant_trials))
    print(f"Trials per participant (should be {n_trials_per_participant}):", set(participant_trials.values()))
    print(f"Total unique combinations: {len(combination_counts)}")
    for count, n_combinations in sorted(stim_summary.items()):
        print(f"{n_combinations} combinations appear {count} times")
    print("Verification complete")
