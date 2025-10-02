# ot-disk-2

Set up a python environment and install the requirements with `pip install -r requirements.txt` 

This uses oTree, which will be installed with the other required packages.

To test it out you simply need to run `otree devserver`

## UV instead of pip / global python

This project uses `uv` to manage the python environment. You can install it easily following the instructions on [https://github.com/astral-sh/uv?tab=readme-ov-file#installation])(the uv github page).

To initially setup the project, run `uv sync` which makes sure all the requirements are installed in the uv environment.

Running the project for testing is done with `uv run otree devserver` which runs the command in the uv environment.
