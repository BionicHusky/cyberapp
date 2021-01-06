import os
import pathlib
import sys

import cv2  # type: ignore
import pyautogui  # type: ignore

from PIL import Image, ImageDraw, ImageFont  # type: ignore

from PySide2.QtCore import QSettings


## Helper function for reading opencv templates as files
def _rt(template_image_path: pathlib.Path):
    return cv2.imread(str(template_image_path), flags=cv2.IMREAD_COLOR)


## Detect if running frozen
if hasattr(sys, "_MEIPASS"):
    MODULE_DIRECTORY = pathlib.Path(sys._MEIPASS).resolve()  # type: ignore
else:
    MODULE_DIRECTORY = pathlib.Path(os.path.realpath(__file__)).parent.resolve()
_version_file = MODULE_DIRECTORY / "VERSION"

## General application constants
APPLICATION_NAME = "cp2077_autohack"
CONFIG_SCHEMA_VERSION = 1
VERSION = (
    _version_file.read_text().strip() if _version_file.is_file() else "development"
)
MAX_SOLUTION_PATH_LENGTH = 16
MAX_POTENTIAL_SOLUTION_THRESHOLD = 20

## Title used to find the process to screenshot
GAME_EXECUTABLE_TITLE = "Cyberpunk 2077 (C) 2020 by CD Projekt RED"

## Directory constants
_qsettings = QSettings(
    QSettings.IniFormat, QSettings.Scope.UserScope, APPLICATION_NAME, application="_"
)
APPLICATION_DIRECTORY = pathlib.Path(_qsettings.fileName()).parent.resolve()

## Make application directory if it doesn't already exist
## Bit of a sketchy place to put this, but oh well
APPLICATION_DIRECTORY.mkdir(parents=True, exist_ok=True)

## Fixed application file/resource paths
CONFIG_FILE_PATH = APPLICATION_DIRECTORY / "config.json"
LOG_FILE_PATH = APPLICATION_DIRECTORY / "log.txt"
RESOURCES_DIRECTORY = MODULE_DIRECTORY / "resources"
IMAGES_DIRECTORY = RESOURCES_DIRECTORY / "images"
FONT_PATH = RESOURCES_DIRECTORY / "fonts/Rajdhani-Regular.ttf"
SEMIBOLD_FONT_PATH = RESOURCES_DIRECTORY / "fonts/Rajdhani-SemiBold.ttf"

## UI constants
LABEL_HINT_STYLE = "color: #444444;"
BUFFER_ERROR_STYLE = "color: #f04e46;"
SELECTION_OFF_COLOR = "#444444"
MAX_IMAGE_SIZE = 400

## Beep constants
BEEP_START = 800
BEEP_SUCCESS = 1000
BEEP_FAIL = 625
BEEP_DURATION = 100

## Breach protocol data constants
CODE_NAMES = ("1C", "55", "7A", "BD", "E9", "FF")

## Preloaded opencv templates
CV_BREACH_TITLE_TEMPLATE = _rt(IMAGES_DIRECTORY / "breach_title.png")
CV_BUFFER_TITLE_TEMPLATE = _rt(IMAGES_DIRECTORY / "buffer_title.png")
CV_SEQUENCES_TITLE_TEMPLATE = _rt(IMAGES_DIRECTORY / "sequences_title.png")
CV_BUFFER_BOX_TEMPLATE = _rt(IMAGES_DIRECTORY / "buffer_box.png")
CV_CODE_TEMPLATES = tuple(
    _rt(IMAGES_DIRECTORY / f"code_{it}.png") for it in range(len(CODE_NAMES))
)
CV_CODE_SMALL_TEMPLATES = tuple(
    _rt(IMAGES_DIRECTORY / f"code_{it}_small.png") for it in range(len(CODE_NAMES))
)
CV_TARGET_TEMPLATES = dict()
_cv_target_templates_raw = {
    "ICEPICK": IMAGES_DIRECTORY / "target_icepick_text.png",
    "DATAMINE_V1": IMAGES_DIRECTORY / "target_datamine_1_text.png",
    "DATAMINE_V2": IMAGES_DIRECTORY / "target_datamine_2_text.png",
    "DATAMINE_V3": IMAGES_DIRECTORY / "target_datamine_3_text.png",
    "MASS VULNERABILITY": IMAGES_DIRECTORY / "target_mass_vulnerability_text.png",
    "CAMERA SHUTDOWN": IMAGES_DIRECTORY / "target_camera_shutdown_text.png",
    "DATAMINE: COPY MALWARE": IMAGES_DIRECTORY / "target_copy_malware_text.png",
    "NEUTRALIZE MALWARE": IMAGES_DIRECTORY / "target_neutralize_malware_text.png",
    "FRIENDLY TURRETS": IMAGES_DIRECTORY / "target_friendly_turrets_text.png",
    "TURRET SHUTDOWN": IMAGES_DIRECTORY / "target_turret_shutdown_text.png",
}
for _target_name, _target_path in _cv_target_templates_raw.items():
    CV_TARGET_TEMPLATES[_target_name] = _rt(_target_path)

## Opencv data parsing constants
ANALYSIS_IMAGE_SIZE = (1920, 1080)
CV_MATRIX_GAP_SIZE = 64.5
CV_BUFFER_BOX_GAP_SIZE = 42.0
CV_SEQUENCES_X_GAP_SIZE = 42.0
CV_SEQUENCES_Y_GAP_SIZE = 70.5
CV_TARGETS_GAP_SIZE = CV_SEQUENCES_Y_GAP_SIZE

## Game window interaction constants
WINDOW_FOCUS_DELAY = 0.4
MOUSE_MOVE_DELAY = 0.01
MOUSE_CLICK_DELAY = 0.01

## Matrix constants
VALID_MATRIX_SIZES = tuple(range(5, 9))
MATRIX_IMAGE_FONT_COLOR = (208, 236, 88, 255)
MATRIX_IMAGE_FONT_SIZE = 30
MATRIX_IMAGE_FONT = ImageFont.truetype(str(SEMIBOLD_FONT_PATH), MATRIX_IMAGE_FONT_SIZE)
MATRIX_IMAGE_SPACING = 15
MATRIX_IMAGE_SIZE = 50
MATRIX_TEMPLATE_HALF_SIZE = 15
MATRIX_COMPOSITE_DISTANCE = MATRIX_IMAGE_SIZE + MATRIX_IMAGE_SPACING

## Generated matrix code images
MATRIX_CODE_IMAGES = list()
for _code_name in CODE_NAMES:
    _code_image = Image.new("RGBA", (MATRIX_IMAGE_SIZE,) * 2, color=(0,) * 4)
    _draw = ImageDraw.Draw(_code_image)
    _draw.text(
        (MATRIX_IMAGE_SIZE / 2,) * 2,
        _code_name,
        anchor="mm",
        font=MATRIX_IMAGE_FONT,
        fill=MATRIX_IMAGE_FONT_COLOR,
    )
    MATRIX_CODE_IMAGES.append(_code_image)

## Matrix sequence path overlay constants
SEQUENCE_PATH_IMAGE_COLOR = (95, 247, 255, 255)
SEQUENCE_PATH_IMAGE_INVALID_COLOR = (240, 78, 70, 255)
SEQUENCE_PATH_IMAGE_BOX_SIZE = 40
SEQUENCE_PATH_IMAGE_THICKNESS = 3
SEQUENCE_PATH_IMAGE_ARROW_SIZE = 10
SEQUENCE_PATH_MAX_SIZE = 8

## Buffer box constants
BUFFER_MIN_X_THRESHOLD = 35
BUFFER_COUNT_THRESHOLD = 8
BUFFER_IMAGE_FONT_SIZE = 22
BUFFER_IMAGE_FONT = ImageFont.truetype(str(SEMIBOLD_FONT_PATH), BUFFER_IMAGE_FONT_SIZE)
BUFFER_IMAGE_SPACING = 8
BUFFER_BOX_IMAGE_COLOR = (118, 135, 50)
BUFFER_BOX_IMAGE_SIZE = 36
BUFFER_BOX_IMAGE_THICKNESS = 2
BUFFER_COMPOSITE_DISTANCE = BUFFER_BOX_IMAGE_SIZE + BUFFER_IMAGE_SPACING
BUFFER_IMAGE_DIMENSIONS = (345, BUFFER_BOX_IMAGE_SIZE)

## Generated buffer boxes and buffer box code images
BUFFER_BOX_IMAGE = Image.new("RGBA", (BUFFER_BOX_IMAGE_SIZE,) * 2, color=(0,) * 4)
_draw = ImageDraw.Draw(BUFFER_BOX_IMAGE)
_draw.rectangle(
    ((0, 0), (BUFFER_BOX_IMAGE_SIZE - 1,) * 2),
    fill=(0,) * 4,
    outline=BUFFER_BOX_IMAGE_COLOR,
    width=BUFFER_BOX_IMAGE_THICKNESS,
)
_quarter_size = int((BUFFER_BOX_IMAGE_SIZE) / 4)
_fifth_size = int((BUFFER_BOX_IMAGE_SIZE) / 5)
_negative_quarter = BUFFER_BOX_IMAGE_SIZE - _quarter_size - 2  ## lol
_lines = (
    ((0, _quarter_size), (BUFFER_BOX_IMAGE_SIZE, _quarter_size)),
    ((0, _negative_quarter), (BUFFER_BOX_IMAGE_SIZE, _negative_quarter)),
)
for _swap in (False, True):
    for _line in _lines:
        _draw.line(
            tuple((_it[::-1] if _swap else _it) for _it in _line),
            fill=(0,) * 4,
            width=_fifth_size,
        )
BUFFER_CODE_IMAGES = list()
for _code_name in CODE_NAMES:
    _code_image = Image.new("RGBA", (BUFFER_BOX_IMAGE_SIZE,) * 2, color=(0,) * 4)
    _draw = ImageDraw.Draw(_code_image)
    _draw.text(
        (BUFFER_BOX_IMAGE_SIZE / 2,) * 2,
        _code_name,
        anchor="mm",
        font=BUFFER_IMAGE_FONT,
        fill=SEQUENCE_PATH_IMAGE_COLOR,
    )
    BUFFER_CODE_IMAGES.append(_code_image)
