"""
A simple script that generates a CSV with a list of all of the stimulus images
used in the experiment.

For each trial, we have:
- target: the reference stimulus
- option_a, option_b: two comparison stimuli

The actual left/right display position is balanced at runtime, not in the CSV.
"""

import csv
from math import ceil
from pathlib import Path
from random import shuffle, seed
# from random import randint
import re


###################################
########## REF VARIABLES ##########
###################################

# DO NOT CHANGE THESE VALUES
# THE FIXED_* VALUES
# ARE JUST FOR REFERENCE AND HAVE NO EFFECT


# How many target discs are there
FIXED_TARGETS: int = 1
# How many comparison options are there
FIXED_OPTIONS: int = 2

###################################
######### MAIN VARIABLES ##########
###################################

# You can change these values to adjust the generation

# this prevents ANY (target or options) stimuli from repeating
# within N trials for a given participant/trial set
# set to 0 to disable this feature
prevent_stimulus_repeat_within_n_trials: int = 2

# Set random seed for reproducibility
random_seed: int = 1999

# number of trial sets to generate
n_participants: int = 155
# expected number of repeats per unique stimuli combination across all trial sets
# this is a minimum target; actual repeats may be higher due to rounding
expected_repeats: int = 3


# Optional: Path to previous trial results CSV to account for completed trials
# if it is None, the script will generate from scratch
# if it is specified, the results from the previous study
# will be subtracted from the expected repeats
previous_trial_results_file: Path | None = Path("./pilot_rated_stims_clean_names_sorted.csv")
# e.g.
# previous_trial_results_file = Path("path_to_pilot/pilot_trial_results.csv")

# Path to the directory containing the stimulus images
stimulus_dir: Path = Path("diskcomp/static/stimuli")

# Path to the CSV file to write the list of stimulus images to
output_file: Path = Path("diskcomp/_private/stim.csv")


###################################
######### MAIN SCRIPT #############
###################################

# set random seed for reproducibility
seed(random_seed)

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

print(f"Generated {n_results} unique combinations") # 23310

# Read previous trial results to account for completed trials
pilot_result_trial_combinations: dict[tuple[str, ...], int] = {}
n_trials_in_pilot = 0

# read csv of existing trials
if previous_trial_results_file is not None:
    if not previous_trial_results_file.exists():
        raise FileNotFoundError(f"Previous trial results file not found: {previous_trial_results_file}")

    with open(previous_trial_results_file, "r", newline="", encoding="utf-8") as f:
        dictReader = csv.DictReader(f)
        for pilot_trial in dictReader:
            stim_a = pilot_trial['option_a']
            stim_b = pilot_trial['option_b']
            target = pilot_trial['target']
            unique_id = tuple(sorted([stim_a, stim_b])) + (target,) # e.g.('93a', '93b', '186b')
            if unique_id not in pilot_result_trial_combinations:
                pilot_result_trial_combinations[unique_id] = 1
            else:
                pilot_result_trial_combinations[unique_id] += 1
            n_trials_in_pilot += 1

print(f"Found {n_trials_in_pilot} completed trials in pilot data")
# show number of combinations that appear more than once from pilot
for combination, count in pilot_result_trial_combinations.items():
    if count > 1:
        print(f"Combination {combination} appears {count} times in pilot data")


# Calculate number of trials per participant
n_trials_per_participant = ceil((n_results * expected_repeats - n_trials_in_pilot) / n_participants)
print(f"To achieve ~{expected_repeats} repeats per combination:"
      f" {n_participants} trial sets x {n_trials_per_participant} trials ="
      f" {n_participants * n_trials_per_participant} total trials")
# 427.935483871
# 66330 -> 66341


shuffle(trial_combinations)
trial_pool: list[tuple[str, str, str]] = []
for _ in range(expected_repeats):
    trial_combinations_copy = trial_combinations.copy()
    shuffle(trial_combinations_copy)
    trial_pool.extend(trial_combinations_copy)

# Create a CSV file that assigns trials to participants
# Note: "participant_id" here represents a trial set, not an actual participant
output_file = Path("diskcomp/_private/trial_list.csv")

print(f"Trial pool size before removing pilot completed trials: {len(trial_pool)}")
# remove trials that were already completed in the pilot
if previous_trial_results_file is not None:
    for completed_combination in pilot_result_trial_combinations.keys():
        # remove the number of completed trials from the trial pool
        option_a, option_b, target = completed_combination
        for _ in range(pilot_result_trial_combinations[completed_combination]):
            if (option_a, option_b, target) in trial_pool:
                trial_pool.remove((option_a, option_b, target))


print(f"Trial pool size after removing pilot completed trials: {len(trial_pool)}")

prevent_stim_repeat: bool = prevent_stimulus_repeat_within_n_trials > 0

expected_total_trials = n_participants * n_trials_per_participant
if len(trial_pool) < expected_total_trials:
    print(f"WARNING: Trial pool size ({len(trial_pool)}) is smaller than expected total trials ({expected_total_trials})!")
    
    number_required = expected_total_trials - len(trial_pool)
    trial_combinations_copy = trial_combinations.copy()
    shuffle(trial_combinations_copy)
    if prevent_stim_repeat:
        print("Adding full set of trials because stimulus repeat prevention is enabled.")
        # add full set of trials to the pool to ensure enough trials
        trial_pool.extend(trial_combinations_copy)
    else:
        print("Randomly allocating extra trials from the full set to fill the gap.")
        trial_pool.extend(trial_combinations_copy[:number_required])


def stims_in_trial_list(targets: tuple[str, str, str], trial_list: list[tuple[str, str, str]]) -> bool:
    all_targets = set(targets)
    all_trial_stims = set([stim for trial in trial_list for stim in trial])
    result = not all_targets.isdisjoint(all_trial_stims)
    if result:
        print(f"Found repeating stim in recent trials: {targets} in {trial_list}")
        pass
    return result


with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["participant_id", "trial_num", "option_a", "option_b", "target", "option_a_file", "option_b_file", "target_file"])
    for p in range(n_participants):
        trials_allocated_to_participant: set[tuple[str, str, str]] = set()
        recent_trials: list[tuple[str, str, str]] = []
        for t in range(n_trials_per_participant):
            extra_index = 0
            option_a, option_b, target = trial_pool[extra_index]
            
            # ensure:
            # 1. no duplicate trials for this participant
            # 2. no stimulus repeats within N trials for this participant
            #    (if configured)
            while (option_a, option_b, target) in trials_allocated_to_participant or \
                  (prevent_stim_repeat and
                   stims_in_trial_list((option_a, option_b, target), recent_trials)):
                extra_index += 1
                extra_index %= len(trial_pool)
                # find a new trial that hasn't been allocated to this participant yet
                option_a, option_b, target = trial_pool[extra_index]
            trials_allocated_to_participant.add((option_a, option_b, target))

            if prevent_stim_repeat:
                recent_trials.append((option_a, option_b, target))
                while len(recent_trials) > prevent_stimulus_repeat_within_n_trials:
                    recent_trials.pop(0)

            trial_pool.remove((option_a, option_b, target))
            writer.writerow([p + 1, t + 1, option_a, option_b, target, stim_file_map[option_a], stim_file_map[option_b], stim_file_map[target]])

print(f"\nCreated trial assignments for {n_participants} trial sets")
print(f"Each trial set has {n_trials_per_participant} trials")

# Verify the assignments
stim_summary: dict[int, int] = {}
stim_summary_excl_pilot: dict[int, int] = {}



with open(output_file, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    participant_trials: dict[str, int] = {}
    combination_counts: dict[tuple[str, ...], int] = pilot_result_trial_combinations.copy()
    # ('93a', '93b', '186b') =/= ('93b', '93a', '186b')
    # this is an example of random number of "completed" tirals from a pilot
    # this would be subtracted from the expected_repeats for those combinations
    # pilot_result_trial_combinations = [
    #     ('93a', '93b', '186b'), # 1
    #     ('93a', '93b', '201a'), # 3
    #     ('93a', '93b', '202a'), # 2
    #     ('93a', '93b', '202b'),
    #     ('93a', '93b', '226a'),
    #     ('93a', '93b', '226b'),
    #     ('93a', '93b', '241a'),
    #     ('93a', '93b', '241b'),
    #     ('93a', '93b', '307a'),
    #     ('93a', '93b', '307b'),
    #     ('93a', '93b', '353a'),
    #     ('93a', '93b', '355a'),
    #     ('93a', '93b', '360a'),
    #     ('93a', '93b', '370a'),
    #     ('93a', '93b', '418a'),
    #     ('93a', '93b', '420a'),
    #     ('93a', '93b', '420b'),
    #     ('93a', '93b', '431a'),
    #     ('93a', '93b', '484a'),
    #     ('93a', '93b', '552a'),
    #     ('93a', '93b', '555a'),
    #     ('93a', '93b', '626a'),
    #     ('93a', '93b', '631a'),
    #     ('93a', '93b', '70a'),
    #     ('93a', '93b', '70b'),
    #     ('93a', '93b', '73a'),
    #     ('93a', '93b', '73b'),
    #     ('93a', '93b', '84a'),
    # ]
    # pilot_result_trial_combinations = {combination: randint(1, 3) for combination in pilot_result_trial_combinations}
    
    for row in reader:
        participant_id = row[0]
        option_a = row[2]
        option_b = row[3]
        target = row[4]

        # Normalize combination order for counting
        combination = tuple(sorted([option_a, option_b])) + (target,)
        if combination not in combination_counts:
            combination_counts[combination] = 1
        else:
            combination_counts[combination] += 1
        
        if participant_id not in participant_trials:
            participant_trials[participant_id] = 1
        else:
            participant_trials[participant_id] += 1
        
        
        
    
    # Check for incorrect trial counts
    for participant_id, n_trials in participant_trials.items():
        if n_trials != n_trials_per_participant:
            print(f"WARNING: Trial set {participant_id} has {n_trials} trials instead of {n_trials_per_participant}")
    
    # Summarize combination frequencies
    for combination, count in combination_counts.items():
        stim_summary[count] = stim_summary.get(count, 0) + 1
        if combination not in pilot_result_trial_combinations:
            stim_summary_excl_pilot[count] = stim_summary_excl_pilot.get(count, 0) + 1
        else:
            # Adjust expected repeats based on pilot data
            pilot_count = pilot_result_trial_combinations[combination]
            adjusted_expected = expected_repeats - pilot_count
            if count < adjusted_expected:
                print(f"WARNING: Combination {combination} appears only {count} times (expected {adjusted_expected} after pilot adjustment)")
            # add remaining counts to the summary
            # subtract pilot count from combination count
            remaining_count = count - pilot_count
            stim_summary_excl_pilot[remaining_count] = stim_summary_excl_pilot.get(remaining_count, 0) + 1
        if count < expected_repeats:  # Allow some flexibility
            print(f"WARNING: Combination {combination} appears only {count} times")
            pass
    
    print("\n=== Verification Summary ===")
    print(f"Number of trial sets: {len(participant_trials)}")
    print(f"Trials per trial set: {set(participant_trials.values())}")
    print(f"Total unique combinations: {len(combination_counts)}")
    print("\nCombination frequency distribution:")
    for count, n_combinations in sorted(stim_summary.items()):
        print(f"  {n_combinations} combinations appear {count} times")
    print("\nCombination frequency distribution (excluding pilot data):")
    for count, n_combinations in sorted(stim_summary_excl_pilot.items()):
        print(f"  {n_combinations} combinations appear {count} times")
    print("\nVerification complete!")
