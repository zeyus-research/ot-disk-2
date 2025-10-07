from sqlalchemy.ext.declarative import DeclarativeMeta  # type: ignore
from otree.api import (  # type: ignore
    BaseConstants,
    BaseSubsession,
    BaseGroup,
    BasePlayer,
    models,
    Page,
    ExtraModel,
)
from otree.models import Participant  # type: ignore
import json
from random import randint, choice
from datetime import datetime, timedelta
from typing import ClassVar, Generator, Any, Literal
from pathlib import Path
import pandas as pd

doc = """
Stimulus comparison study with trial set pooling
"""


class AnnotationFreeMeta(DeclarativeMeta):
    """Metaclass to remove the __annotations__ attribute from the class
    this fixes an error where oTree tries to use __annotations__ and thinks it's a dict
    that needs saving.
    """

    def __new__(cls, name, bases, dct):
        dct.pop("__annotations__", None)
        return super().__new__(cls, name, bases, dct)


class C(BaseConstants):
    NAME_IN_URL: str = "diskcomp"
    PLAYERS_PER_GROUP: int | None = None
    NUM_ROUNDS: int = 1
    STIM_PATH: Path = Path("stimuli")
    STIM_CSV: Path = Path(__file__).parent / "_private/trial_list.csv"
    STIM_IMAGE_CSV: Path = Path(__file__).parent / "_private/stim.csv"
    NUM_PRACTICE_TRIALS: int = 3
    TRIALS_IN_BLOCK: int = 100
    TOTAL_TRIALS: int = 400
    NUM_BLOCKS: int = 4

    # Primary timeout: minimum time before a trial set can be considered for abandonment
    TRIAL_SET_TIMEOUT_MINUTES: int = 60

    # Inactivity timeout: if no response for this many minutes, consider participant inactive
    # Trial set is only abandoned if BOTH conditions are met:
    # 1. Time since lock > TRIAL_SET_TIMEOUT_MINUTES
    # 2. Time since last response > INACTIVITY_TIMEOUT_MINUTES
    INACTIVITY_TIMEOUT_MINUTES: int = 5

    # Optional: List of specific trial set IDs to load
    # If empty, all trial sets from CSV will be loaded
    TRIAL_SETS_TO_LOAD: list[int] = []


class DataCache:
    cache: pd.DataFrame | None = None
    image_file_list: list[str] | None = None

    @classmethod
    def get(cls) -> pd.DataFrame:
        if cls.cache is None:
            cls.cache = pd.read_csv(C.STIM_CSV)
        return cls.cache
    
    @classmethod
    def get_image_file_list(cls) -> list[str]:
        if cls.image_file_list is None:
            image_df = pd.read_csv(C.STIM_IMAGE_CSV)
            cls.image_file_list = image_df["filename"].map(lambda x: (C.STIM_PATH / x).as_posix()).tolist()
        return cls.image_file_list

class Subsession(BaseSubsession, metaclass=AnnotationFreeMeta):
    pass

class Group(BaseGroup, metaclass=AnnotationFreeMeta):
    pass


class TrialSet(ExtraModel, metaclass=AnnotationFreeMeta):
    """Represents a set of trials (originally mapped to a participant_id in CSV)"""
    set_id: int = models.IntegerField()
    repeat_id: int = models.IntegerField(initial=0)  # 0 for original, 1+ for retries after timeout
    locked_by_participant: str = models.StringField(blank=True)
    lock_time: float = models.FloatField(initial=0.0)
    last_response_time: float = models.FloatField(initial=0.0)  # Updated each time participant responds
    completed: bool = models.BooleanField(initial=False)
    abandoned: bool = models.BooleanField(initial=False)  # True if timed out/incomplete
    subsession: BaseSubsession = models.Link(Subsession)
    current_trial: int = models.IntegerField(initial=0)
    participant: Participant | None = models.Link(Participant)
    
    @classmethod
    def get_available_set(cls, subsession: BaseSubsession) -> 'TrialSet | None':
        """Find an available trial set (unlocked, timed out, or not completed)"""
        all_sets = cls.filter(subsession=subsession, completed=False, abandoned=False)
        timeout_seconds = C.TRIAL_SET_TIMEOUT_MINUTES * 60
        inactivity_seconds = C.INACTIVITY_TIMEOUT_MINUTES * 60
        current_time = datetime.now().timestamp()

        for trial_set in all_sets:
            # Check if unlocked
            if not trial_set.locked_by_participant:
                return trial_set

            # Check if timed out - requires BOTH conditions:
            # 1. Total time since lock exceeds primary timeout
            # 2. No response activity within inactivity timeout
            if trial_set.lock_time > 0:
                time_since_lock = current_time - trial_set.lock_time

                # Check primary timeout
                if time_since_lock > timeout_seconds:
                    # Check inactivity timeout
                    # Use last_response_time if set, otherwise fall back to lock_time
                    last_activity = trial_set.last_response_time if trial_set.last_response_time > 0 else trial_set.lock_time
                    time_since_last_response = current_time - last_activity

                    if time_since_last_response > inactivity_seconds:
                        # Both conditions met - mark as abandoned and create retry
                        trial_set.abandoned = True
                        trial_set.locked_by_participant = ""

                        print(f"Abandoning trial set {trial_set.set_id} (repeat {trial_set.repeat_id}): "
                              f"locked for {time_since_lock/60:.1f}min, "
                              f"inactive for {time_since_last_response/60:.1f}min")

                        # Create a new trial set with incremented repeat_id
                        new_trial_set = cls._create_retry_trial_set(subsession, trial_set)
                        return new_trial_set

        return None

    @classmethod
    def _create_retry_trial_set(cls, subsession: BaseSubsession, original_set: 'TrialSet') -> 'TrialSet':
        """Create a copy of a trial set for retry after timeout"""
        # Create new trial set with incremented repeat_id
        new_trial_set = cls.create(
            set_id=original_set.set_id,
            repeat_id=original_set.repeat_id + 1,
            subsession=subsession,
        )

        # Copy all trials from the original set to the new set
        original_trials = Trial.filter(trial_set=original_set)
        for original_trial in original_trials:
            Trial.create(
                trial_id=original_trial.trial_id,
                target=original_trial.target,
                option_a=original_trial.option_a,
                option_b=original_trial.option_b,
                csv_order=original_trial.csv_order,
                trial_set=new_trial_set,
            )

        print(f"Created retry trial set: set_id={new_trial_set.set_id}, repeat_id={new_trial_set.repeat_id}")
        return new_trial_set
    
    def lock_for_participant(self, participant: Participant):
        """Lock this trial set for a specific participant"""
        self.locked_by_participant = participant.code
        self.participant = participant
        self.lock_time = datetime.now().timestamp()
    
    def unlock(self):
        """Release the lock on this trial set"""
        # noop, it is marked as completed or abandoned elsewhere


class Player(BasePlayer, metaclass=AnnotationFreeMeta):
    assigned_trial_set_id: int = models.IntegerField(initial=-1)
    num_trials: int = models.IntegerField(initial=C.TOTAL_TRIALS)
    
    def get_assigned_trial_set(self) -> TrialSet | None:
        """Get the trial set assigned to this player"""
        if self.assigned_trial_set_id == -1:
            return None
        sets = TrialSet.filter(
            subsession=self.subsession,
            set_id=self.assigned_trial_set_id
        )
        return sets[0] if sets else None
    
    def assign_trial_set(self) -> TrialSet | None:
        """Assign an available trial set to this player"""
        if self.assigned_trial_set_id != -1:
            # Already assigned
            return self.get_assigned_trial_set()
        
        trial_set = TrialSet.get_available_set(self.subsession)
        if trial_set:
            trial_set.lock_for_participant(self.participant)
            self.assigned_trial_set_id = trial_set.set_id
            return trial_set
        
        return None


class Trial(ExtraModel, metaclass=AnnotationFreeMeta):
    trial_id: int = models.IntegerField()  # 0-399 for each trial in the set
    target: str = models.StringField()
    option_a: str = models.StringField()  # First comparison stimulus
    option_b: str = models.StringField()  # Second comparison stimulus
    csv_order: int = models.IntegerField()
    trial_set: TrialSet = models.Link(TrialSet)
    trial_start: float = models.FloatField(initial=0.0)
    trial_end: float = models.FloatField(initial=0.0)
    response: str = models.StringField(initial="")  # 'option_a' or 'option_b'
    displayed_left: str = models.StringField(initial="")  # Which option was shown on left
    response_time: float = models.FloatField(initial=0.0)
    participant_code: str = models.StringField(blank=True)




def get_practice_stims() -> list[str]:
    stims = C.STIM_PATH / "practice"
    return [
        (stims / "practice1.jpg").as_posix(),
        (stims / "practice2.jpg").as_posix(),
        (stims / "practice3.jpg").as_posix(),
        (stims / "practice4.jpg").as_posix(),
        (stims / "practice5.jpg").as_posix(),
        (stims / "practice6.jpg").as_posix(),
    ]


def practice_trial_generator() -> Generator[dict[str, str | int | bool], None, None]:
    stims = get_practice_stims()
    n_practice = len(stims)
    for i in range(C.NUM_PRACTICE_TRIALS):
        target_idx = randint(0, n_practice - 1)
        remaining = [j for j in range(n_practice) if j != target_idx]
        left_idx = choice(remaining)
        remaining.remove(left_idx)
        right_idx = choice(remaining)

        yield {
            "target": stims[target_idx],
            "left_option": stims[left_idx],
            "right_option": stims[right_idx],
            "trial": i,
        }


def creating_session(subsession: Subsession) -> None:
    """Initialize trial sets and trials from CSV"""
    # Read the full CSV
    stims = DataCache.get()
    
    # Determine which trial sets to load
    if C.TRIAL_SETS_TO_LOAD:
        trial_set_ids = C.TRIAL_SETS_TO_LOAD
        stims = stims[stims["participant_id"].isin(trial_set_ids)]
    else:
        trial_set_ids = stims["participant_id"].unique().tolist()
    
    print(f"Loading {len(trial_set_ids)} trial sets")
    
    # Create TrialSet entities
    for set_id in trial_set_ids:
        TrialSet.create(
            set_id=set_id,
            subsession=subsession,
        )
    
    # Create Trial entities for each trial set
    for set_id in trial_set_ids:
        trial_set = TrialSet.filter(subsession=subsession, set_id=set_id)[0]
        set_trials = stims[stims["participant_id"] == set_id]
        
        for _, row in set_trials.iterrows():
            # Store both options as option_a and option_b
            # The actual left/right display will be determined at runtime
            Trial.create(
                trial_id=row["trial_num"] - 1,  # 0-indexed
                target=(C.STIM_PATH / row["target_file"]).as_posix(),
                option_a=(C.STIM_PATH / row["option_a_file"]).as_posix(),
                option_b=(C.STIM_PATH / row["option_b_file"]).as_posix(),
                csv_order=set_id,
                trial_set=trial_set,
            )
    
    for p in subsession.get_players():
        p.num_trials = C.TOTAL_TRIALS


def get_current_trial(player: Player, *, update_start_time: bool = True) -> Trial | None:
    """Get the current trial for a player's assigned trial set"""
    trial_set = player.get_assigned_trial_set()
    if not trial_set:
        return None
    
    trials = Trial.filter(trial_set=trial_set, trial_id=trial_set.current_trial)
    if not trials:
        return None
    
    t = trials[0]
    if update_start_time and t.trial_start == 0.0:
        t.trial_start = datetime.now().timestamp()
    return t


def determine_display_positions(trial: Trial) -> tuple[str, str, str]:
    """
    Determine left/right display positions for a trial.
    Returns (left_option_path, right_option_path, displayed_left_name)
    
    Balancing strategy: alternate by trial number to ensure each
    option appears equally on left and right sides.
    """
    if trial.trial_id % 2 == 0:
        return (trial.option_a, trial.option_b, "option_a")
    else:
        return (trial.option_b, trial.option_a, "option_b")


# PAGES
class WelcomePage(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1


class StimuliComparisonPage(Page):
    block_id: ClassVar[int] = 1
    
    @staticmethod
    def is_displayed(player: Player):
        if player.assigned_trial_set_id == -1:
            trial_set = player.assign_trial_set()
            if not trial_set:
                return False
        
        trial_set = player.get_assigned_trial_set()
        if not trial_set:
            return False
        
        return trial_set.current_trial < C.TRIALS_IN_BLOCK

    @staticmethod
    def vars_for_template(player: Player):
        trial = get_current_trial(player)
        if not trial:
            return {
                "trial_id": 0,
                "target": "",
                "left_option": "",
                "right_option": "",
                "num_trials": C.TOTAL_TRIALS,
                "trials_in_block": C.TRIALS_IN_BLOCK,
                "page_type": "experiment",
                "image_preloads": [],
            }

        left_option, right_option, displayed_left = determine_display_positions(trial)
        trial.displayed_left = displayed_left
        return {
            "trial_id": trial.trial_id,
            "target": trial.target,
            "left_option": left_option,
            "right_option": right_option,
            "num_trials": C.TOTAL_TRIALS,
            "trials_in_block": C.TRIALS_IN_BLOCK,
            "page_type": "experiment",
            "image_preloads": DataCache.get_image_file_list(),
        }

    @staticmethod
    def live_method(player: Player, data: dict[str, Any], block: Literal[1, 2, 3, 4] = 1):
        response: dict[str, Any] = {
            player.id_in_group: {
                "event": "error",
                "page_type": "experiment",
            }
        }
        
        trial_set = player.get_assigned_trial_set()
        if not trial_set:
            response[player.id_in_group]["event"] = "error"
            response[player.id_in_group]["message"] = "No trial set assigned"
            return response
        
        if "event" in data:
            event = data["event"]

            if event == "start":
                response[player.id_in_group]["event"] = "start_ack"
                trial = get_current_trial(player)
                if trial:
                    trial.trial_start = datetime.now().timestamp()
                    
            elif event == "choice":
                trial = get_current_trial(player, update_start_time=False)
                if trial and trial.trial_id == int(data["trial"]):
                    # Record trial response data
                    current_time = datetime.now().timestamp()

                    # Convert 'left' or 'right' to 'option_a' or 'option_b'
                    if data["choice"] == "left":
                        chosen_option = trial.displayed_left
                    else:  # 'right'
                        chosen_option = "option_a" if trial.displayed_left == "option_b" else "option_b"

                    # Set all fields at once to ensure they're persisted together
                    trial.response = chosen_option
                    trial.trial_end = current_time
                    trial.response_time = current_time - trial.trial_start
                    trial.participant_code = player.participant.code

                    # Update trial set's last response time to track activity
                    trial_set.last_response_time = current_time

                    print(f"Trial {trial_set.current_trial + 1} / {C.TOTAL_TRIALS}, "
                          f"RT: {trial.response_time:.2f}s, Response: {trial.response}, "
                          f"Saved: response={trial.response}, rt={trial.response_time}")

                    trial_set.current_trial += 1

                    if trial_set.current_trial < block * C.TRIALS_IN_BLOCK:
                        response[player.id_in_group]["event"] = "next"
                    else:
                        response[player.id_in_group]["event"] = "end"

            elif event == "next":
                trial = get_current_trial(player)
                if trial:
                    left_option, right_option, displayed_left = determine_display_positions(trial)
                    trial.displayed_left = displayed_left
                    
                    response[player.id_in_group]["event"] = "trial"
                    response[player.id_in_group]["trial_id"] = trial.trial_id
                    response[player.id_in_group]["target"] = trial.target
                    response[player.id_in_group]["left_option"] = left_option
                    response[player.id_in_group]["right_option"] = right_option

        return response


class StimuliComparisonPageBlockTwo(StimuliComparisonPage):
    template_name: str = "diskcomp/StimuliComparisonPage.html"
    block_id: ClassVar[int] = 2
    
    @staticmethod
    def is_displayed(player: Player):
        trial_set = player.get_assigned_trial_set()
        if not trial_set:
            return False
        return (trial_set.current_trial >= C.TRIALS_IN_BLOCK and 
                trial_set.current_trial < 2 * C.TRIALS_IN_BLOCK)
    
    @staticmethod
    def live_method(player: Player, data: dict[str, Any], _: Literal[1, 2, 3, 4] = 2):
        return StimuliComparisonPage.live_method(player, data, block=2)


class StimuliComparisonPageBlockThree(StimuliComparisonPage):
    template_name: str = "diskcomp/StimuliComparisonPage.html"
    block_id: ClassVar[int] = 3
    
    @staticmethod
    def is_displayed(player: Player):
        trial_set = player.get_assigned_trial_set()
        if not trial_set:
            return False
        return (trial_set.current_trial >= 2 * C.TRIALS_IN_BLOCK and 
                trial_set.current_trial < 3 * C.TRIALS_IN_BLOCK)
    
    @staticmethod
    def live_method(player: Player, data: dict[str, Any], _: Literal[1, 2, 3, 4] = 3):
        return StimuliComparisonPage.live_method(player, data, block=3)


class StimuliComparisonPageBlockFour(StimuliComparisonPage):
    template_name: str = "diskcomp/StimuliComparisonPage.html"
    block_id: ClassVar[int] = 4
    
    @staticmethod
    def is_displayed(player: Player):
        trial_set = player.get_assigned_trial_set()
        if not trial_set:
            return False
        return (trial_set.current_trial >= 3 * C.TRIALS_IN_BLOCK and 
                trial_set.current_trial < 4 * C.TRIALS_IN_BLOCK)
    
    @staticmethod
    def live_method(player: Player, data: dict[str, Any], _: Literal[1, 2, 3, 4] = 4):
        return StimuliComparisonPage.live_method(player, data, block=4)


class BreakPage(Page):
    pass


class BreakPageTwo(BreakPage):
    template_name: str = "diskcomp/BreakPage.html"


class BreakPageThree(BreakPage):
    template_name: str = "diskcomp/BreakPage.html"


class PracticeInstructionsPage(Page):
    pass


class PracticeDone(Page):
    pass


class ThankYouPage(Page):
    @staticmethod
    def is_displayed(player: Player):
        if player.round_number != C.NUM_ROUNDS:
            return False
        
        trial_set = player.get_assigned_trial_set()
        if trial_set:
            trial_set.completed = True
            trial_set.unlock()
        
        return True


class PracticeTrialPage(Page):
    template_name: str = "diskcomp/StimuliComparisonPage.html"

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1

    @staticmethod
    def vars_for_template(player: Player):
        practice_trials = practice_trial_generator()
        trial = next(practice_trials)
        return {
            "trial_id": trial["trial"],
            "target": trial["target"],
            "left_option": trial["left_option"],
            "right_option": trial["right_option"],
            "num_trials": C.NUM_PRACTICE_TRIALS,
            "trials_in_block": C.NUM_PRACTICE_TRIALS,
            "page_type": "practice",
            "image_preloads": DataCache.get_image_file_list(),
        }

    @staticmethod
    def live_method(player: Player, data: dict[str, Any]):
        response: dict[str, Any] = {
            player.id_in_group: {
                "event": "error",
                "page_type": "practice",
            }
        }

        if "event" in data:
            event = data["event"]

            if event == "start":
                practice_trials = practice_trial_generator()
                trial = next(practice_trials)
                response[player.id_in_group]["event"] = "trial"
                response[player.id_in_group]["trial_id"] = trial["trial"]
                response[player.id_in_group]["target"] = trial["target"]
                response[player.id_in_group]["left_option"] = trial["left_option"]
                response[player.id_in_group]["right_option"] = trial["right_option"]
                
            elif event == "choice":
                if int(data["trial"]) < C.NUM_PRACTICE_TRIALS - 1:
                    response[player.id_in_group]["trial_id"] = data["trial"]
                    response[player.id_in_group]["event"] = "next"
                else:
                    response[player.id_in_group]["event"] = "end"

            elif event == "next":
                practice_trials = practice_trial_generator()
                trial = next(practice_trials)
                if "trial" in data:
                    response[player.id_in_group]["trial_id"] = int(data["trial"]) + 1
                else:
                    response[player.id_in_group]["trial_id"] = trial["trial"]
                response[player.id_in_group]["event"] = "trial"
                response[player.id_in_group]["target"] = trial["target"]
                response[player.id_in_group]["left_option"] = trial["left_option"]
                response[player.id_in_group]["right_option"] = trial["right_option"]

        return response


page_sequence = [
    WelcomePage,
    PracticeInstructionsPage,
    PracticeTrialPage,
    PracticeDone,
    StimuliComparisonPage,
    BreakPage,
    StimuliComparisonPageBlockTwo,
    BreakPageTwo,
    StimuliComparisonPageBlockThree,
    BreakPageThree,
    StimuliComparisonPageBlockFour,
    ThankYouPage,
]


def extract_stimulus_id(file_path: str) -> str:
    """Extract stimulus ID from file path (e.g., 'stimuli/93b_...' -> '93b')"""
    if not file_path:
        return ""
    filename = Path(file_path).name
    # Extract the ID (first part before underscore or extension)
    return filename.split("_")[0].split(".")[0]


def custom_export(players: list[Player]) -> Generator[list[str | int | float | bool | str], Any, Any]:
    """Export trial data with trial set information - includes all trials even from incomplete sessions"""
    yield [
        "participant_code",
        "participant_label",
        "trial_set_id",
        "repeat_id",
        "trial_set_completed",
        "trial_set_abandoned",
        "trial_id",
        "target_stimulus",
        "option_a_stimulus",
        "option_b_stimulus",
        "displayed_left_stimulus",
        "displayed_right_stimulus",
        "response_stimulus",
        "response_time",
    ]

    # Export ALL trial sets (including abandoned ones) to preserve all data
    all_trial_sets = TrialSet.filter()
    for trial_set in all_trial_sets:
        for trial in Trial.filter(trial_set=trial_set):
            # Only export trials that have been answered (have a response recorded)
            if not trial.response or not trial.participant_code:
                continue

            participant_code = trial.participant_code
            participant_label = trial_set.participant.label if trial_set.participant else "UNKNOWN"

            # Extract stimulus IDs from file paths
            target_id = extract_stimulus_id(trial.target)
            option_a_id = extract_stimulus_id(trial.option_a)
            option_b_id = extract_stimulus_id(trial.option_b)

            # Determine which stimulus was displayed on left/right
            if trial.displayed_left == "option_a":
                displayed_left = option_a_id
                displayed_right = option_b_id
            else:
                displayed_left = option_b_id
                displayed_right = option_a_id

            # Convert response to stimulus ID
            if trial.response == "option_a":
                response_stimulus = option_a_id
            elif trial.response == "option_b":
                response_stimulus = option_b_id
            else:
                response_stimulus = trial.response

            yield [
                participant_code,
                participant_label,
                trial_set.set_id,
                trial_set.repeat_id,
                trial_set.completed,
                trial_set.abandoned,
                trial.trial_id,
                target_id,
                option_a_id,
                option_b_id,
                displayed_left,
                displayed_right,
                response_stimulus,
                trial.response_time,
            ]
