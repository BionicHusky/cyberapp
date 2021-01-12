import dataclasses
import traceback

from typing import Iterable, List, Optional, Tuple

import numpy  # type: ignore
import pyautogui  # type: ignore
import pydantic
import system_hotkey  # type: ignore

from . import constants
from . import exceptions
from .logger import LOG


class Config(pydantic.BaseSettings):
    schema_version: int = pydantic.Field(
        constants.CONFIG_SCHEMA_VERSION, description="Config version number"
    )
    show_autohack_warning_message: bool = pydantic.Field(
        True, description="Show the autohack warning message to not move the mouse"
    )
    auto_autohack: bool = pydantic.Field(
        False,
        description="Run autohacking automatically if all daemons can be obtained",
    )
    enable_beeps: bool = pydantic.Field(
        False, description="Allow for beep notifications"
    )
    analysis_hotkey: List[str] = pydantic.Field(
        ["control", "shift", "h"], description="Hotkey for kicking off analysis"
    )
    game_focus_delay: int = pydantic.Field(
        400,
        description="Amount of milliseconds to wait after the game is focused",
        ge=50,
        le=10000,
    )
    buffer_size_override: int = pydantic.Field(
        0, description="Buffer size manual override", ge=0, le=10
    )
    matrix_code_detection_threshold: float = pydantic.Field(
        0.8,
        description="Detection threshold for codes in the matrix",
        gt=0.0,
        lt=1.0,
    )
    buffer_box_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for buffer boxes",
        gt=0.0,
        lt=1.0,
    )
    sequence_code_detection_threshold: float = pydantic.Field(
        0.7,
        description="Detection threshold for codes in the sequences",
        gt=0.0,
        lt=1.0,
    )
    daemon_detection_threshold: float = pydantic.Field(
        0.8,
        description="Detection threshold for sequence daemon names",
        gt=0.0,
        lt=1.0,
    )
    core_detection_threshold: float = pydantic.Field(
        0.8,
        description=(
            "Detection threshold for core elements of the breach protocol screen"
        ),
        gt=0.0,
        lt=1.0,
    )
    detection_language: str = pydantic.Field(
        "English", description="Breach protocol screen language for analysis"
    )
    force_autohack: bool = pydantic.Field(
        False, description="Changes analysis to always present a solvable solution"
    )
    daemon_skip_priorities: List[constants.Daemon] = pydantic.Field(
        list(), description="Daemons to skip first during a forced autohack"
    )
    autohack_activation_key: str = pydantic.Field(
        "f", description="Key to press for autohacking activations"
    )
    autohack_keypress_delay: int = pydantic.Field(
        17,
        description="Milisecond delay between keypresses during the autohack",
        ge=10,
        le=100,
    )

    @pydantic.validator("analysis_hotkey")
    def hotkey_sequence_validator(cls, value: List[str]) -> List[str]:
        valid = set(system_hotkey.vk_codes).union(set(system_hotkey.win_modders))
        invalid_keys = set(value).difference(valid)
        if invalid_keys:
            raise ValueError(
                f"The following keys are invalid: {', '.join(invalid_keys)}"
            )

        if value:
            hotkey = system_hotkey.SystemHotkey()
            try:
                hotkey.parse_hotkeylist(value)
            except Exception as exception:
                LOG.exception("analysis_hotkey sequence invalid:")
                raise ValueError(f"Analysis hotkey sequnece invalid: {exception}")
        return value

    @pydantic.validator("detection_language")
    def detection_language_validator(cls, value: str) -> str:
        if value not in constants.TEMPLATE_LANGUAGE_DATA:
            raise ValueError(f"Detection language {value} is not supported")
        return value

    @pydantic.validator("autohack_activation_key")
    def autohack_activation_key_validator(cls, value: str) -> str:
        if not value:
            raise ValueError("Key must be defined")
        if value not in pyautogui.KEY_NAMES:
            raise ValueError(f"{value} is not a valid key")
        return value


@dataclasses.dataclass
class ScreenshotData:
    screenshot: numpy.ndarray


@dataclasses.dataclass
class BreachProtocolData:
    data: List[List[int]]
    matrix_size: int
    buffer_size: int
    sequences: List[List[int]]
    daemons: List[Optional[constants.Daemon]]
    daemon_names: List[str]


@dataclasses.dataclass
class ScreenBounds:
    code_matrix: Tuple[Tuple[int, int], Tuple[int, int]]
    buffer_box: Tuple[Tuple[int, int], Tuple[int, int]]
    sequences: Tuple[Tuple[int, int], Tuple[int, int]]
    daemons: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclasses.dataclass
class AnalysisData:
    breach_protocol_data: BreachProtocolData
    screenshot_data: ScreenshotData
    screen_bounds: ScreenBounds


@dataclasses.dataclass
class SequencePathData:
    all_sequence_paths: Tuple[Tuple[int, ...], ...]
    shortest_solution: Optional[Tuple[int, ...]]
    shortest_solution_path: Optional[Tuple[Tuple[int, int], ...]]
    solution_valid: bool
    computationally_complex: bool


@dataclasses.dataclass(eq=True, frozen=True)
class Sequence:
    string: str
    contiguous_block_indices: Tuple[int, ...]


@dataclasses.dataclass
class ConvertedSequence:
    data: Tuple[int, ...]
    contiguous_block_indices: Tuple[int, ...]


@dataclasses.dataclass
class SequenceSelectionData:
    daemon: Optional[constants.Daemon]
    daemon_name: str
    sequence: Tuple[int, ...]
    selected: bool


class Error:
    """Exception container with additional information"""

    def __init__(self, exception: Exception):
        message = str(exception)
        traceback_string = ""
        if isinstance(exception, exceptions.CPAHException):
            critical = exception.critical
            unhandled = False
        else:
            traceback_string = "\n".join(traceback.format_tb(exception.__traceback__))
            unhandled = critical = True

        if unhandled:
            message = (
                f"An unhandled error occurred ({exception.__class__.__name__}):"
                f"\n{message}\n\n{traceback_string}"
            )

        self.exception = exception
        self.traceback = traceback_string
        self.unhandled = unhandled
        self.critical = critical
        self.message = message
        self.title = "Error"

    def __str__(self):
        return (
            "<Error "
            f"exception=<{self.exception}>, "
            f'traceback="{self.traceback}", '
            f"unhandled={self.unhandled}, "
            f"critical={self.critical}, "
            f'message="{self.message}">'
        )
