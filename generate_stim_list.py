"""
A simple script that generates a CSV with a list of all of the stimulus images
used in the experiment.
"""

import csv
from pathlib import Path
import re

# Path to the directory containing the stimulus images
stimulus_dir = Path("disks/static/stim")

# Path to the CSV file to write the list of stimulus images to
output_file = Path("disks/_private/stim.csv")

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
