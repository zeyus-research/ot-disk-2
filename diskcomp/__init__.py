from sqlalchemy.ext.declarative import DeclarativeMeta  # type: ignore
from otree.api import (  # type: ignore
    BaseConstants,
    BaseSubsession,
    BaseGroup,
    BasePlayer,
    models,
    Page,
    ExtraModel,
    widgets,
)
from otree.models import Participant  # type: ignore
import json
from random import randint, choice
from datetime import datetime
from typing import ClassVar, Generator, Any, Literal
from pathlib import Path
import pandas as pd

doc = """
Stimulus comparison study
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
    NUM_ROUNDS: int = 1  # one for each exp type, using live pages
    STIM_PATH: Path = Path("stimuli")
    STIM_CSV: Path = Path(__file__).parent / "_private/trial_list.csv"
    NUM_PRACTICE_TRIALS: int = 3
    TRIALS_IN_BLOCK: int = 2


class DataCache:
    cache: pd.DataFrame | None = None

    @classmethod
    def get(cls) -> pd.DataFrame:
        if cls.cache is None:
            # read the CSV file
            # Modify this based on your CSV structure
            # Example format: ID,order,trial,target,left_option,right_option,correct_option
            cls.cache = pd.read_csv(C.STIM_CSV)
        return cls.cache


class Group(BaseGroup, metaclass=AnnotationFreeMeta):
    pass


class Player(BasePlayer, metaclass=AnnotationFreeMeta):
    trial_id: int = models.IntegerField(initial=0)
    num_trials: int = models.IntegerField()


class Trial(ExtraModel, metaclass=AnnotationFreeMeta):
    trial_id: int = models.IntegerField()
    target: str = models.StringField()
    left_option: str = models.StringField()
    right_option: str = models.StringField()
    correct_option: str = models.StringField()  # 'left' or 'right'
    csv_order: int = models.IntegerField()
    player: Player = models.Link(Player)
    trial_start: float = models.FloatField(initial=0.0)
    trial_end: float = models.FloatField(initial=0.0)
    response: str = models.StringField(initial="")  # 'left' or 'right'
    correct: bool = models.BooleanField(initial=False)
    response_time: float = models.FloatField(initial=0.0)


class Subsession(BaseSubsession, metaclass=AnnotationFreeMeta):
    pass


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
        # Select a target (always different from options)
        target_idx = randint(0, n_practice - 1)
        
        # Select left and right options (different from target and each other)
        remaining = [j for j in range(n_practice) if j != target_idx]
        left_idx = choice(remaining)
        remaining.remove(left_idx)
        right_idx = choice(remaining)
        
        # Randomly select which option is correct
        correct_option = 'left' if randint(0, 1) == 0 else 'right'

        yield {
            "target": stims[target_idx],
            "left_option": stims[left_idx],
            "right_option": stims[right_idx],
            "correct_option": correct_option,
            "trial": i,
        }


def get_stim_list(id: int) -> pd.DataFrame:
    return DataCache.get()
    # # ID = participant ID, Paradigm = 0 or 1 = order
    # stims = DataCache.get()
    # # get the stim list for this player
    # stim_list = stims[(stims["ID"] == id)]
    # return stim_list


def creating_session(subsession: Subsession) -> None:
    # the order column represents the experiment type (0 or 1)
    for i, p in enumerate(subsession.get_players()):
        # get the stim list for this player
        stim_list = get_stim_list(i)
        num_trials = len(stim_list)
        p.trial_id = 0
        p.num_trials = num_trials
        # save the stim order for this player
        for _, row in stim_list.iterrows():
            Trial.create(
                trial_id=row["trial"],
                target=(C.STIM_PATH / row["target"]).with_suffix(".png").as_posix(),
                left_option=(C.STIM_PATH / row["left_option"]).with_suffix(".png").as_posix(),
                right_option=(C.STIM_PATH / row["right_option"]).with_suffix(".png").as_posix(),
                correct_option=row["correct_option"],  # 'left' or 'right'
                csv_order=row["order"],
                player=p,
            )


def get_current_trial(player: Player, *, update_start_time: bool = True) -> Trial:
    t = Trial.filter(player=player, trial_id=player.trial_id)[0]
    if update_start_time and t.trial_start == 0.0:
        t.trial_start = datetime.now().timestamp()
    return t


# PAGES
class WelcomePage(Page):
    # only display this page on the first round
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == 1


class StimuliComparisonPage(Page):
    block_id: ClassVar[int] = 1
    # only display this page on the first round
    @staticmethod
    def is_displayed(player: Player):
        return player.trial_id < player.num_trials

    @staticmethod
    def vars_for_template(player: Player):
        trial = get_current_trial(player)

        return {
            "trial_id": trial.trial_id,
            "target": trial.target,
            "left_option": trial.left_option,
            "right_option": trial.right_option,
            "num_trials": player.num_trials,
            "trials_in_block": C.TRIALS_IN_BLOCK,
            "page_type": "experiment",
        }

    @staticmethod
    def live_method(player: Player, data: dict[str, Any], block: Literal[1, 2, 3] = 1):
        # prepare response (default to error)
        response: dict[str, Any] = {
            player.id_in_group: {
                "event": "error",
                "page_type": "experiment",
            }
        }
        # must have an event key
        if "event" in data:
            event = data["event"]

            if event == "start":
                response[player.id_in_group]["event"] = "start_ack"
                trial = get_current_trial(player)
                trial.trial_start = datetime.now().timestamp()
            # the participant has made a selection
            elif event == "choice":
                trial = get_current_trial(player, update_start_time=False)
                if trial.trial_id == int(data["trial"]):
                    trial.response = data["choice"]  # 'left' or 'right'
                    # Check if the participant selected the correct option
                    trial.correct = (data["choice"] == trial.correct_option)
                    trial.trial_end = datetime.now().timestamp()
                    trial.response_time = trial.trial_end - trial.trial_start
                    print(player.trial_id + 1, " / ", player.num_trials)
                    print("response time: ", trial.response_time)
                    player.trial_id += 1
                    if player.trial_id < block * C.TRIALS_IN_BLOCK:
                        response[player.id_in_group]["event"] = "next"
                    else:
                        response[player.id_in_group]["event"] = "end"

            # the page requests the next stimulus
            elif event == "next":
                trial = get_current_trial(player)
                response[player.id_in_group]["event"] = "trial"
                response[player.id_in_group]["trial_id"] = trial.trial_id
                response[player.id_in_group]["target"] = trial.target
                response[player.id_in_group]["left_option"] = trial.left_option
                response[player.id_in_group]["right_option"] = trial.right_option

        return response
    

class StimuliComparisonPageBlockTwo(StimuliComparisonPage):
    template_name: str = "diskcomp/StimuliComparisonPage.html"
    block_id: ClassVar[int] = 2
    @staticmethod
    def live_method(player: Player, data: dict[str, Any], block: Literal[1, 2, 3] = 2):
        return StimuliComparisonPage.live_method(player, data, block)


class StimuliComparisonPageBlockThree(StimuliComparisonPage):
    template_name: str = "diskcomp/StimuliComparisonPage.html"
    block_id: ClassVar[int] = 3
    @staticmethod
    def live_method(player: Player, data: dict[str, Any], block: Literal[1, 2, 3] = 3):
        return StimuliComparisonPage.live_method(player, data, block)

class BreakPage(Page):
    pass

class BreakPageTwo(BreakPage):
    template_name: str = "diskcomp/BreakPage.html"

class PracticeInstructionsPage(Page):
    pass


class PracticeDone(Page):
    pass


class ThankYouPage(Page):
    # only display this page on the last round
    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == C.NUM_ROUNDS

class PracticeTrialPage(Page):
    template_name: str = "diskcomp/StimuliComparisonPage.html"

    # only display this page on the first round
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
        }

    @staticmethod
    def live_method(player: Player, data: dict[str, Any]):
        # prepare response (default to error)
        response: dict[str, Any] = {
            player.id_in_group: {
                "event": "error",
                "page_type": "practice",
            }
        }

        # must have an event key
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
                response[player.id_in_group]["correct_option"] = trial["correct_option"]
            # the participant has made a selection
            elif event == "choice":
                # Store the response (left or right)
                choice = data["choice"]
                # Get the correct option from trial data
                practice_trials = practice_trial_generator()
                trial_data = next(practice_trials)
                for _ in range(int(data["trial"])):
                    trial_data = next(practice_trials)
                    
                # Check if correct and proceed
                correct = (choice == trial_data["correct_option"])
                
                if int(data["trial"]) < C.NUM_PRACTICE_TRIALS - 1:
                    response[player.id_in_group]["trial_id"] = data["trial"]
                    response[player.id_in_group]["event"] = "next"
                else:
                    response[player.id_in_group]["event"] = "end"

            # the page requests the next stimulus
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
                response[player.id_in_group]["correct_option"] = trial["correct_option"]

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
    ThankYouPage,
]


def custom_export(players: list[Player]) -> Generator[list[str | int | float | bool], Any, Any]:
    # header row
    yield [
        "participant_code",
        "participant_label",
        "trial_id",
        "target",
        "left_option",
        "right_option",
        "correct_option",
        "csv_order",
        "response",
        "correct",
        "response_time",
        "id_in_group",
        "id_in_csv",
    ]
    for player in players:
        for trial in Trial.filter(player=player):
            yield [
                player.participant.code,
                player.participant.label or "",
                trial.trial_id,
                trial.target,
                trial.left_option,
                trial.right_option,
                trial.correct_option,
                trial.csv_order,
                trial.response,
                trial.correct,
                trial.response_time,
                player.id_in_group,
                player.id_in_group - 1,
            ]
