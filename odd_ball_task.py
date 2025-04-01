############################################################################
####################### The Oddball Experiment #############################
############### The Portable Disc - Social Network Study ###################
#################### @Kristian Tylén  - eSYMb 2024 #########################
############################################################################

# import libraries
from psychopy import event, core, visual, data
from psychopy.gui import wxgui as gui
import random
import pandas as pd
import os, glob
from itertools import combinations

# define dialogue box (important that this happens before you define window)
box = gui.Dlg(title = "The disc oddball experiment")
box.addField("Participant ID: ") 
box.addField("Age: ")
box.addField("Gender: ", choices=["Female", "Male", "Other" ])
box.addField("Nationality: ")
box.show()
if box.OK: # To retrieve data from popup window
    ID = box.data[0]
    AGE = box.data[1]
    GENDER = box.data[2]
    NATIONALITY = box.data[3]
elif box.Cancel: # To cancel the experiment if popup is closed
    core.quit()

# Define window
win = visual.Window(fullscr = False, size = [1200, 700], color = "white", units='height')

# Define stopwatch
clock = core.Clock()

# Define mouse
mouse = event.Mouse()

# get date for unique logfile name
date = data.getDateStr()

# define logfile 
columns = ['time_stamp', 'id', 'age', 'gender', 'nationality', 'oddball', 'foil', 'oddball_position', 'accuracy', 'reaction_time'] 
logfile = pd.DataFrame(columns=columns)

# make sure that there is a logfile directory and otherwise make one
if not os.path.exists("logfiles"):
    os.makedirs("logfiles")

# define logfile name
logfile_name = "logfiles/logfile_{}_{}.csv".format(ID, date)

### Instruction texts 

instruction1 = '''
Thank you for participating in this experiment. At every trial, you will see six decorated discs.\n 
5 out of 6 are exactly the same; only one shape is different; this is the "oddball".\n 
You have to find the oddball and click on it.\n\n\n 
Press space to continue...'''

instruction2 = '''
You can take time to prepare, but once you click on the fixation cross to start the trial,\n 
you have to respond as fast as possible! Both time and the precision matter.\n\n
You will be given feedback after each trial.\n\n
We will start with a practice session consisting of three trials.\n\n\n
Press space when you are ready...'''

instruction3 = '''
That was it for the practice trials - good job!\n
Now follows the actual experiment.\n\n\n
Press space to start...'''

goodbye = '''
The experiment is done.\n 
Thank you for your participation.'''

# fixation cross
fix = visual.TextStim(win, text = "+", height = 0.04, color = "black")
fix_box = visual.Rect(win, height = 0.05, width = 0.06, fillColor = None, lineColor = "black")

# text
txt = visual.TextStim(win, text = None, height = 0.02, color = "black", alignHoriz='left', pos = (-0.5, 0)) 
trial_counter = visual.TextStim(win, text = None, height = 0.02, color = "black", alignHoriz='left', pos = (0.55, -0.4))

# define directories
STIMULUS_DIR = 'stimuli/'
STIMULUS_idx = len(STIMULUS_DIR)

# get stimulus images
stimuli_a = glob.glob(STIMULUS_DIR + '*.png')
#stimuli_b = glob.glob(STIMULUS_DIR + '*b.png')

# combine and randomize
STIM_COMBI = list(combinations(stimuli_a, 2))
#STIM_COMBI_b = list(combinations(stimuli_b, 2))
#STIM_COMBI = STIM_COMBI_a + STIM_COMBI_b

# Define objects and positions
stimulus1 = visual.ImageStim(win, image = None, size = 0.15, pos = (-.11,.165))
stimulus2 = visual.ImageStim(win, image = None, size = 0.15, pos = (.11,.165))
stimulus3 = visual.ImageStim(win, image = None, size = 0.15, pos = (-.2,0))
stimulus4 = visual.ImageStim(win, image = None, size = 0.15, pos = (.2,0))
stimulus5 = visual.ImageStim(win, image = None, size = 0.15, pos = (-.11,-.165))
stimulus6 = visual.ImageStim(win, image = None, size = 0.15, pos = (.11,-.165))
stim_list = [stimulus1, stimulus2, stimulus3, stimulus4, stimulus5, stimulus6]

# feedback icons
success = "success.png"
failure = "failure.png"
feedback = visual.ImageStim(win, image = None, size = 0.1) 

# Stimulus dictionaries
trial_list = []
for stim in STIM_COMBI:
    # make it random what will be oddball and what will be foil
    random.shuffle([stim[0],stim[1]])
    oddball = stim[0]
    foil = stim[1]
    trial_list +=[{
    'oddball': oddball,
    'foil': foil}]

# Randomize trials 
random.shuffle(trial_list)

#### FUNCTIONS #####

# function that prepare a trial
def prepare_stim(index):
    
    oddball = trial_list[index]['oddball']
    foil = trial_list[index]['foil']
    
    random.shuffle(stim_list)
    
    trial_stim = []
    for s in range(len(stim_list)):
        if s == 0:
            stim_list[s].image = oddball
            stype = "oddball"
            oddball_pos = stim_list[s].pos
        else:
            stim_list[s].image = foil
            stype = "foil"
        
        trial_stim += [{
            'stimulus': stim_list[s],
            'oddball_image': oddball,
            'foil_image': foil,
            'type': stype,
            'pos': stim_list[s].pos}]
    
    return trial_stim, oddball, foil, oddball_pos

def draw_stim():
    for trial in trial_stim:
        trial['stimulus'].draw()
    feedback.draw()
    win.flip()

# draw text function
def msg(text):
    txt.text = text
    txt.draw()
    win.flip()
    event.waitKeys()

def draw_fixation(x,y):
    count = "number of trials: {}/{}".format(x+1,y+3)
    trial_counter.text = count
    feedback.image = None
    
    while True:
        fix_box.draw()
        fix.draw()
        trial_counter.draw()
        win.flip()
        
        # get mouse button presses
        mouse1, mouse2, mouse3 = mouse.getPressed()
        if mouse.isPressedIn(fix_box):
            win.flip()
            core.wait(0.2)
            break

#### RUN EXPERIMENT ####

response = None

msg(instruction1)
msg(instruction2)

for i in range(len(trial_list)):
    
    # prepare stimulus set for current trial
    trial_stim, oddball, foil, oddball_pos = prepare_stim(i)
    
    # draw fixation
    draw_fixation(i,len(trial_list))
    
    # reset stopwatch  
    clock.reset()
    
    end_trial = False
    while not end_trial:
        
        # reset mouse
        mouse_down_detected = False
    
        # get mouse button presses
        mouse1, mouse2, mouse3 = mouse.getPressed()
        
        # draw shapes
        draw_stim()
        
        # record key presses
        key = event.getKeys(keyList = ["escape"])
        # if escape is pressed - quit the experiment 
        if key == ["escape"]:
            #  save the logfile
            logfile.to_csv(logfile_name)
            # say goodbye
            msg(goodbye)
            # quit the experiment
            core.quit()
        
        # recond mouse presses
            
        for t in trial_stim:
            if mouse.isPressedIn(t['stimulus']):
                rt = clock.getTime()
                
                if t['type'] == "oddball":
                    feedback.image = success
                    response = 1
                
                elif t['type'] == "foil":     
                    feedback.image = failure
                    response = 0
                feedback.pos = t['pos']
                
                # draw shapes
                draw_stim()
                
                # write logfile 
                logfile = logfile._append({
                    'time_stamp': date,
                    'id': ID,
                    'age': AGE,
                    'gender': GENDER,
                    'nationality': NATIONALITY,
                    'oddball': oddball,
                    'foil': foil,
                    'oddball_position': oddball_pos, 
                    'accuracy': response,
                    'reaction_time': rt}, ignore_index = True)
                
                # save data to directory
                logfile.to_csv(logfile_name)
                
                core.wait(0.5)
                end_trial = True

msg(goodbye)