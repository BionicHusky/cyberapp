import copy
import itertools
import json
import pathlib
import sys
import time

from typing import Iterable, List, Optional, Set, Tuple

import cv2  # type: ignore
import numpy  # type: ignore
import pyautogui  # type: ignore
import win32gui  # type: ignore

from PIL import Image, ImageDraw  # type: ignore
from PySide2.QtCore import Signal
from PySide2.QtGui import QPixmap

from . import constants
from . import exceptions
from . import models
from .mouse_helper import MOUSE
from .logger import LOG


def load_config() -> models.Config:
    LOG.debug(f"Loading config file from {constants.CONFIG_FILE_PATH}")
    if not constants.CONFIG_FILE_PATH.is_file():
        LOG.info("No config file found. Loading all defaults")
        return models.Config()
    with constants.CONFIG_FILE_PATH.open("r") as config_file:
        try:
            config_data = json.load(config_file)
            ## NOTE: Change if config scheme is updated
            config_version = config_data["schema_version"]
            assert (
                config_version == constants.CONFIG_SCHEMA_VERSION
            ), f"Unknown config version {config_version}"
            config = models.Config(**config_data)
        except Exception as exception:
            LOG.exception(
                f"Failed to load the config file at {constants.CONFIG_FILE_PATH}"
            )
            raise exceptions.CPAHInvalidConfigException(str(exception))
    LOG.debug("Config file loaded")
    return config


def save_config(config: models.Config):
    LOG.debug(f"Saving config file to {constants.CONFIG_FILE_PATH}")
    constants.CONFIG_FILE_PATH.write_text(config.json(indent=4))
    LOG.debug("Config file saved")


def convert_code(code: Iterable[int]) -> Tuple[str, ...]:
    """Helper function to convert a list of code IDs to strings."""
    return tuple(constants.CODE_NAMES[it] for it in code)


def generate_matrix_image(image_size: int, matrix_data: List[List[int]]) -> Image.Image:
    """Generates a matrix image of codes to be displayed in the GUI."""
    matrix_image = Image.new("RGBA", (image_size,) * 2, color=(0, 0, 0, 0))
    for column_index, column_data in enumerate(matrix_data):
        for row_index, code in enumerate(column_data):
            matrix_image.alpha_composite(
                constants.MATRIX_CODE_IMAGES[code],
                dest=(
                    constants.MATRIX_COMPOSITE_DISTANCE * row_index,
                    constants.MATRIX_COMPOSITE_DISTANCE * column_index,
                ),
            )
    if image_size > constants.MAX_IMAGE_SIZE:
        matrix_image = matrix_image.resize((constants.MAX_IMAGE_SIZE,) * 2)
    return matrix_image


def generate_sequence_path_image(
    image_size: int, sequence_path: Tuple[Tuple[int, int], ...], valid_path: bool
) -> Image.Image:
    sequence_path_image = Image.new("RGBA", (image_size,) * 2, color=(0, 0, 0, 0))
    offset = int(constants.MATRIX_IMAGE_SIZE / 2) - 1  ## lol
    box_offset = int(constants.SEQUENCE_PATH_IMAGE_BOX_SIZE / 2)
    draw = ImageDraw.Draw(sequence_path_image)

    if valid_path:
        color = constants.SEQUENCE_PATH_IMAGE_COLOR
    else:
        color = constants.SEQUENCE_PATH_IMAGE_INVALID_COLOR

    ## Draw all lines in the path
    converted_path = list()
    for coordinate in sequence_path:
        converted_path.append(
            (
                offset + coordinate[0] * constants.MATRIX_COMPOSITE_DISTANCE,
                offset + coordinate[1] * constants.MATRIX_COMPOSITE_DISTANCE,
            )
        )
    draw.line(converted_path, fill=color, width=constants.SEQUENCE_PATH_IMAGE_THICKNESS)

    ## Draw all boxes and arrows
    spias = constants.SEQUENCE_PATH_IMAGE_ARROW_SIZE
    for coordinate_index, coordinate in enumerate(converted_path):
        box_coordinates = (
            (coordinate[0] - box_offset, coordinate[1] - box_offset),
            (coordinate[0] + box_offset, coordinate[1] + box_offset),
        )
        draw.rectangle(
            box_coordinates,
            fill=(0, 0, 0, 0),
            outline=color,
            width=constants.SEQUENCE_PATH_IMAGE_THICKNESS,
        )

        ## Draw arrows on all boxes except the last one
        if coordinate_index < len(converted_path) - 1:
            next_coordinate = converted_path[coordinate_index + 1]
            if next_coordinate[1] == coordinate[1]:  ## Same row
                direction = 1 if next_coordinate[0] > coordinate[0] else -1
                swap = False
            else:  ## Same column
                direction = 1 if next_coordinate[1] > coordinate[1] else -1
                swap = True
            x_coordinates = (
                coordinate[swap] + box_offset * direction,
                coordinate[swap] + box_offset * direction,
                coordinate[swap] + (box_offset + spias) * direction,
            )
            y_coordinates = (
                coordinate[not swap] - box_offset,
                coordinate[not swap] + box_offset,
                coordinate[not swap],
            )
            if swap:
                coordinates_zip = tuple(zip(y_coordinates, x_coordinates))
            else:
                coordinates_zip = tuple(zip(x_coordinates, y_coordinates))
            draw.polygon(coordinates_zip, fill=color)

    if image_size > constants.MAX_IMAGE_SIZE:
        sequence_path_image = sequence_path_image.resize(
            (constants.MAX_IMAGE_SIZE,) * 2
        )
    return sequence_path_image


def generate_buffer_boxes_image(buffer_size: int) -> Image.Image:
    """Creates the buffer boxes base image."""
    buffer_boxes_image = Image.new(
        "RGBA", constants.BUFFER_IMAGE_DIMENSIONS, color=(0, 0, 0, 0)
    )

    for box_index in range(buffer_size):
        buffer_boxes_image.alpha_composite(
            constants.BUFFER_BOX_IMAGE,
            dest=(constants.BUFFER_COMPOSITE_DISTANCE * box_index, 0),
        )

    return buffer_boxes_image


def generate_buffer_sequence_image(sequence: Tuple[int, ...]) -> Image.Image:
    """Creates the sequence of codes for the buffer."""
    buffer_sequence_image = Image.new(
        "RGBA", constants.BUFFER_IMAGE_DIMENSIONS, color=(0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(buffer_sequence_image)
    for index, code in enumerate(sequence):
        x_offset = constants.BUFFER_COMPOSITE_DISTANCE * index
        rectangle_corner = constants.BUFFER_BOX_IMAGE_SIZE - 1
        draw.rectangle(
            ((x_offset, 0), (x_offset + rectangle_corner, rectangle_corner)),
            fill=(0,) * 4,
            outline=constants.SEQUENCE_PATH_IMAGE_COLOR,
            width=constants.BUFFER_BOX_IMAGE_THICKNESS,
        )
        buffer_sequence_image.alpha_composite(
            constants.BUFFER_CODE_IMAGES[code], dest=(x_offset, 0)
        )

    return buffer_sequence_image


def _focus_game_window() -> int:
    """Helper function to focus the game window."""
    LOG.debug("Bringing CP2077 to the foreground")
    hwnd = win32gui.FindWindow(None, constants.GAME_EXECUTABLE_TITLE)
    if hwnd == win32gui.GetForegroundWindow():
        LOG.debug("Game window already active")
        return hwnd
    if not hwnd:
        raise exceptions.CPAHGameNotFoundException()
    win32gui.SetForegroundWindow(hwnd)
    time.sleep(constants.WINDOW_FOCUS_DELAY)
    return hwnd


def grab_screenshot(
    config: models.Config, from_file: Optional[pathlib.Path] = None
) -> models.ScreenshotData:
    """Takes a screenshot of the game and calculates some stuff."""
    if from_file:
        LOG.debug(f"Loading screenshot from file: {from_file}")
        screenshot = Image.open(from_file)
        x_start = y_start = 0
        x_end, y_end = screenshot.size
    else:
        LOG.debug(f"Taking a screenshot from the game")
        hwnd = _focus_game_window()
        x_start, y_start, x_end, y_end = win32gui.GetWindowRect(hwnd)
        LOG.debug(f"WindowRect: ({x_start}, {y_start}), ({x_end}, {y_end})")
        if x_start != 0 or y_start != 0:  ## Hacky way to check if not in fullscreen
            LOG.debug("CP2077 detected to be running in windowed mode")
            x_start, y_start, x_end, y_end = win32gui.GetClientRect(hwnd)
            LOG.debug(f"ClientRect: ({x_start}, {y_start}), ({x_end}, {y_end})")
            x_start, y_start = win32gui.ClientToScreen(hwnd, (x_start, y_start))
            x_end, y_end = win32gui.ClientToScreen(
                hwnd, (x_end - x_start, y_end - y_start)
            )
            LOG.debug(f"ClientToScreen: ({x_start}, {y_start}), ({x_end}, {y_end})")

        ## Move mouse to the bottom of the screen to avoid the cursor getting in the way
        MOUSE.move((x_end, y_end))

        screenshot = pyautogui.screenshot(region=(x_start, y_start, x_end, y_end))

    original_resolution = screenshot.size
    LOG.debug(f"Screenshot resolution: {original_resolution}")

    ## Detect and account for resolution aspect ratios other than 16:9
    for dimension_index in (0, 1):
        order = 1 if dimension_index else -1
        half_position = screenshot.size[dimension_index] / 2
        pixel_counter = 0
        black_bar_pixel = screenshot.getpixel((pixel_counter, half_position)[::order])
        while black_bar_pixel == (0, 0, 0):
            pixel_counter += 1
            black_bar_pixel = screenshot.getpixel(
                (pixel_counter, half_position)[::order]
            )

        if pixel_counter > 0:
            LOG.info(
                "Cropping and resizing incorrect aspect ratio screenshot. "
                f"Black bar size: {pixel_counter}"
            )
            crop_start = (pixel_counter, 0)[::order]
            crop_end = (
                screenshot.size[not dimension_index] - pixel_counter,
                screenshot.size[dimension_index],
            )[::order]
            screenshot = screenshot.crop(crop_start + crop_end)

    screenshot = screenshot.resize(constants.ANALYSIS_IMAGE_SIZE)

    ## Convert screenshot to numpy array (so cv2 can use them)
    screenshot = numpy.array(screenshot)[:, :, ::-1].copy()
    screenshot_data = models.ScreenshotData(screenshot=screenshot)
    return screenshot_data


def parse_screen_bounds(
    config: models.Config, screenshot_data: models.ScreenshotData
) -> models.ScreenBounds:
    """Finds key parts of the breach protocol screen and records them."""
    ## Search for the breach title ("BREACH TIME REMAINING")
    breach_title_results = cv2.matchTemplate(
        screenshot_data.screenshot,
        constants.CV_BREACH_TITLE_TEMPLATE,
        cv2.TM_CCOEFF_NORMED,
    )
    _, confidence, _, location = cv2.minMaxLoc(breach_title_results)
    if confidence < config.core_detection_threshold:
        raise exceptions.CPAHScreenshotParseFailedException(
            "Confidence for the breach title (core) match is below threshold: "
            f"{confidence:.3f} < {config.core_detection_threshold}"
        )

    ## Assign coordinate elements
    code_matrix_x_top = location[0]

    ## Search for the buffer title ("BUFFER")
    buffer_title_results = cv2.matchTemplate(
        screenshot_data.screenshot,
        constants.CV_BUFFER_TITLE_TEMPLATE,
        cv2.TM_CCOEFF_NORMED,
    )
    _, confidence, _, location = cv2.minMaxLoc(buffer_title_results)
    if confidence < config.core_detection_threshold:
        raise exceptions.CPAHScreenshotParseFailedException(
            "Confidence for the buffer title (core) match is below threshold: "
            f"{confidence:.3f} < {config.core_detection_threshold}"
        )

    ## Assign coordinate elements
    height = constants.CV_BUFFER_TITLE_TEMPLATE.shape[1::-1][1]
    code_matrix_x_bottom = sequences_x_top = buffer_x_top = location[0]
    buffer_y_top = location[1] + height

    ## Search for the sequence title ("SEQUENCE REQUIRED TO UPLOAD")
    sequence_title_results = cv2.matchTemplate(
        screenshot_data.screenshot,
        constants.CV_SEQUENCES_TITLE_TEMPLATE,
        cv2.TM_CCOEFF_NORMED,
    )
    _, confidence, _, location = cv2.minMaxLoc(sequence_title_results)
    if confidence < config.core_detection_threshold:
        raise exceptions.CPAHScreenshotParseFailedException(
            "Confidence for the sequence title (core) match is below threshold: "
            f"{confidence:.3f} < {config.core_detection_threshold}"
        )

    ## Assign coordinate elements
    width, height = constants.CV_SEQUENCES_TITLE_TEMPLATE.shape[1::-1]
    buffer_y_bottom = location[1]
    code_matrix_y_top = sequences_y_top = targets_y_top = location[1] + height
    targets_x_top = sequences_x_bottom = location[0] + width

    ## Estimates for the rest of the search bounds
    bottom_cutoff = code_matrix_y_top + int(constants.ANALYSIS_IMAGE_SIZE[1] * 0.5)
    code_matrix_y_bottom = sequences_y_bottom = targets_y_bottom = bottom_cutoff
    targets_x_bottom = buffer_x_bottom = int(constants.ANALYSIS_IMAGE_SIZE[0] * 0.85)

    return models.ScreenBounds(
        code_matrix=(
            (code_matrix_x_top, code_matrix_y_top),
            (code_matrix_x_bottom, code_matrix_y_bottom),
        ),
        buffer_box=(
            (buffer_x_top, buffer_y_top),
            (buffer_x_bottom, buffer_y_bottom),
        ),
        sequences=(
            (sequences_x_top, sequences_y_top),
            (sequences_x_bottom, sequences_y_bottom),
        ),
        targets=(
            (targets_x_top, targets_y_top),
            (targets_x_bottom, targets_y_bottom),
        ),
    )


def parse_matrix_data(
    config: models.Config,
    screenshot_data: models.ScreenshotData,
    screen_bounds: models.ScreenBounds,
) -> List[List[int]]:
    """Parses the section of the screenshot to get matrix data."""
    box = screen_bounds.code_matrix
    crop = screenshot_data.screenshot[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    ## Search for each code
    x_max = y_max = 0
    x_min = y_min = 9999
    all_code_points = dict()
    for index, template in enumerate(constants.CV_CODE_TEMPLATES):
        code_results = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        matches = numpy.where(code_results >= config.matrix_code_detection_threshold)
        all_code_points[index] = code_points = list(zip(*matches[::-1]))
        for point in code_points:
            if point[0] > x_max:
                x_max = point[0]
            if point[0] < x_min:
                x_min = point[0]
            if point[1] > y_max:
                y_max = point[1]
            if point[1] < y_min:
                y_min = point[1]

    LOG.debug(f"Matrix parsing found min/max: ({x_min}, {y_min}) / ({x_max}, {y_max})")

    ## Adjust code points based on maximum and minimum values
    matrix_size = round((x_max - x_min) / constants.CV_MATRIX_GAP_SIZE) + 1
    if matrix_size not in constants.VALID_MATRIX_SIZES:
        raise exceptions.CPAHMatrixParseFailedException(
            f"Detected matrix size is invalid ({matrix_size})."
        )

    ## Yes, it would be easier to use a numpy 2d array. But I'm dumb.
    data: List[List[int]] = [
        [-1 for _ in range(matrix_size)] for _ in range(matrix_size)
    ]
    for code_index, points in all_code_points.items():
        for x, y in points:
            row = round((x - x_min) / constants.CV_MATRIX_GAP_SIZE)
            column = round((y - y_min) / constants.CV_MATRIX_GAP_SIZE)
            data[column][row] = code_index

    LOG.debug(f"Matrix parsing found data: {data}")

    for check_row in data:
        for check_it in check_row:
            if check_it == -1:
                raise exceptions.CPAHMatrixParseFailedException(
                    "Matrix code detection could not detect all codes."
                )

    return data


def parse_buffer_size_data(
    config: models.Config,
    screenshot_data: models.ScreenshotData,
    screen_bounds: models.ScreenBounds,
) -> int:
    """Parses the section of the screenshot to get buffer size data."""

    if config.buffer_size_override > 0:
        LOG.info(f"Buffer size override from config: {config.buffer_size_override}")
        return config.buffer_size_override

    box = screen_bounds.buffer_box
    crop = screenshot_data.screenshot[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    ## Search for each box
    x_max = 0
    x_min = 9999
    buffer_box_results = cv2.matchTemplate(
        crop, constants.CV_BUFFER_BOX_TEMPLATE, cv2.TM_CCOEFF_NORMED
    )
    matches = numpy.where(buffer_box_results >= config.buffer_box_detection_threshold)
    buffer_box_points = list(zip(*matches[::-1]))
    LOG.debug(f"Buffer box parsing found matches: {buffer_box_points}")
    for point in buffer_box_points:
        if point[0] < x_min:
            x_min = point[0]
        if point[0] > x_max:
            x_max = point[0]

    LOG.debug(f"Buffer box parsing found x min/max: {x_min} / {x_max}")

    if x_min > constants.BUFFER_MIN_X_THRESHOLD:
        LOG.error(
            "Buffer box start position is past the start threshold "
            f"({x_min} > {constants.BUFFER_MIN_X_THRESHOLD})"
        )
        raise exceptions.CPAHBufferParseFailedException(
            "Buffer box detection is misaligned."
        )

    buffer_size = round((x_max - x_min) / constants.CV_BUFFER_BOX_GAP_SIZE) + 1

    LOG.debug(f"Buffer box parsing found buffer size: {buffer_size}")
    if not 0 < buffer_size <= constants.BUFFER_COUNT_THRESHOLD:
        raise exceptions.CPAHBufferParseFailedException(
            f"Invalid buffer size detected ({buffer_size})."
        )
    return buffer_size


def parse_sequences_data(
    config: models.Config,
    screenshot_data: models.ScreenshotData,
    screen_bounds: models.ScreenBounds,
) -> List[List[int]]:
    """Parses the section of the screenshot to get sequences data."""
    box = screen_bounds.sequences
    crop = screenshot_data.screenshot[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    ## Search for each code
    y_max = 0
    x_min = y_min = 9999
    all_sequence_points = dict()
    for index, template in enumerate(constants.CV_CODE_SMALL_TEMPLATES):
        code_results = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        matches = numpy.where(code_results >= config.sequence_code_detection_threshold)
        all_sequence_points[index] = code_points = list(zip(*matches[::-1]))
        for point in code_points:
            if point[0] < x_min:
                x_min = point[0]
            if point[1] > y_max:
                y_max = point[1]
            if point[1] < y_min:
                y_min = point[1]

    LOG.debug(f"Sequence parsing found min/max: ({x_min}, {y_min}) / ({y_max})")

    sequences_size = round((y_max - y_min) / constants.CV_SEQUENCES_Y_GAP_SIZE) + 1
    data: List[List[int]] = [[] for _ in range(sequences_size)]
    for code_index, points in all_sequence_points.items():
        for x, y in points:
            sequence_index = round((y - y_min) / constants.CV_SEQUENCES_Y_GAP_SIZE)
            code_position = round((x - x_min) / constants.CV_SEQUENCES_X_GAP_SIZE)
            sequence = data[sequence_index]
            size_difference = code_position - len(sequence) + 1
            if size_difference > 0:
                sequence.extend([-1] * size_difference)
            sequence[code_position] = code_index

    LOG.debug(f"Sequence parsing found data: {data}")

    for sequence_index, sequence in enumerate(data):
        if len(sequence) > constants.SEQUENCE_PATH_MAX_SIZE:
            raise exceptions.CPAHSequenceParseFailedException(
                f"Sequence {sequence_index + 1} read to be too long ({len(sequence)})."
            )
        for code in sequence:
            if code == -1:
                raise exceptions.CPAHSequenceParseFailedException(
                    f"Sequence {sequence_index + 1} could not be parsed correctly."
                )

    return data


def parse_targets_data(
    config: models.Config,
    screenshot_data: models.ScreenshotData,
    screen_bounds: models.ScreenBounds,
    sequences_size: int,
) -> List[str]:
    """Parses the section of the screenshot to get targets data."""
    box = screen_bounds.targets
    crop = screenshot_data.screenshot[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    ## Search for each target
    y_max = 0
    x_min = y_min = 9999
    all_target_points = dict()
    for target_name, template in constants.CV_TARGET_TEMPLATES.items():
        target_results = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, location = cv2.minMaxLoc(target_results)
        if confidence >= config.target_detection_threshold:
            all_target_points[target_name] = location
            if location[0] < x_min:
                x_min = location[0]
            if location[1] > y_max:
                y_max = location[1]
            if location[1] < y_min:
                y_min = location[1]

    LOG.debug(f"Target parsing found min/max: ({x_min}, {y_min}) / ({y_max})")

    ## NOTE: There's a bug here. If the first target is unknown, y_min is incorrect
    targets_size = round((y_max - y_min) / constants.CV_SEQUENCES_Y_GAP_SIZE) + 1
    data = ["UNKNOWN" for _ in range(sequences_size)]
    for target_name, point in all_target_points.items():
        target_index = round((point[1] - y_min) / constants.CV_SEQUENCES_Y_GAP_SIZE)
        data[target_index] = target_name

    LOG.debug(f"Target parsing found data: {data}")
    return data


def calculate_sequence_path_data(
    breach_protocol_data: models.BreachProtocolData,
    selected_sequence_indices: Optional[Tuple[int, ...]] = None,
) -> models.SequencePathData:
    ## Please dear god don't let there be a bug with the solver.
    ## I will 100% not remember how this works if I need to come back and debug it.

    ## Convert sequences to a list of strings to support sublist searching
    ## NOTE: this means that the solver will break if there are more than 10 code types
    base_sequences = [
        "".join(str(it) for it in seq) for seq in breach_protocol_data.sequences
    ]

    potential_solutions: Set[models.Sequence] = set()

    def _recursive_build(
        sequence: models.Sequence, remaining_permutation: Tuple[int, ...]
    ):
        """Recursively adds potential solutions given the permutation."""
        nonlocal potential_solutions

        ## Sequences that are too long are not discarded because
        ## required buffer size (shortest solution) must be calculated.
        if not remaining_permutation:
            potential_solutions.add(sequence)
            return

        other_sequence = base_sequences[remaining_permutation[0]]

        ## If the entire other_sequence is already found in the sequence, skip the rest
        if other_sequence in sequence.string:
            ## Different continuous blocks can occur for matching substrings
            intersection = False
            search_range = range(len(sequence.string) - len(other_sequence) + 1)
            for start_index in search_range:
                ## If the location of other_sequence in the current sequence intersects
                ## the boundary between two contiguous blocks, erase that boundary
                ## to make a larger combined contiguous block.
                if sequence.string[start_index:].startswith(other_sequence):
                    new_sequence = sequence
                    for it in range(len(other_sequence)):
                        test_block_index = start_index + 1 + it
                        if test_block_index in sequence.contiguous_block_indices:
                            temp_list = list(new_sequence.contiguous_block_indices)
                            temp_list.remove(test_block_index)
                            new_sequence = models.Sequence(
                                string=new_sequence.string,
                                contiguous_block_indices=tuple(temp_list),
                            )
                    _recursive_build(
                        new_sequence, remaining_permutation=remaining_permutation[1:]
                    )
            return

        ## Search substrings to see if any parts match
        for substring_index in range(1, len(other_sequence)):
            other_sequence_substring = other_sequence[:-substring_index]
            if sequence.string.endswith(other_sequence_substring):
                new_sequence = models.Sequence(
                    string=sequence.string + other_sequence[-substring_index:],
                    contiguous_block_indices=sequence.contiguous_block_indices,
                )
                _recursive_build(new_sequence, remaining_permutation[1:])

        ## Current sequence does not end with other sequence. Start a new contiguous block
        new_sequence = models.Sequence(
            string=sequence.string + other_sequence,
            contiguous_block_indices=(
                sequence.contiguous_block_indices + (len(sequence.string),)
            ),
        )
        _recursive_build(new_sequence, remaining_permutation[1:])

    permutations = itertools.permutations(
        selected_sequence_indices or range(len(base_sequences))
    )
    for current_permutation in permutations:
        _recursive_build(
            models.Sequence(string="", contiguous_block_indices=tuple()),
            current_permutation,
        )

    ## i'm leaving this in for the meme
    sorted_converted_potential_solutions: Tuple[models.ConvertedSequence, ...] = tuple(
        models.ConvertedSequence(
            data=tuple(int(it) for it in sequence.string),
            contiguous_block_indices=sequence.contiguous_block_indices,
        )
        for sequence in sorted(potential_solutions, key=lambda x: len(x.string))
    )

    ## Too many potential solutions to calculate, skip trying to find the shortest
    ## solution if the solution is invalid
    skip_invalid_immediately = (
        len(potential_solutions) > constants.MAX_POTENTIAL_SOLUTION_THRESHOLD
    )
    if skip_invalid_immediately:
        LOG.warning(
            f"Too many potential solutions found ({len(potential_solutions)}), "
            "will skip trying to find shortest invalid solution."
        )

    ## Lexically define before _recursive_solve to keep mypy happy
    current_solutions: List[Tuple[Tuple[int, int], ...]] = list()

    def _recursive_solve(
        path: Tuple[Tuple[int, int], ...],
        in_row: bool,
        sequence: models.ConvertedSequence,
        sequence_index: int,
        explore_count: int,
    ):
        """
        path: coordinate path through the matrix that solves the sequence
        in_row: search row or column for values
        remaining: remaining code sequences to obtain
        explore_count: current number of wasted moves used to look for the next target
        """
        nonlocal current_solutions

        ## Throw away solutions that are waaaaaayyy too long
        if skip_invalid_immediately or len(path) > constants.MAX_SOLUTION_PATH_LENGTH:
            return

        ## End of sequence reached, add to current solutions
        if sequence_index == len(sequence.data):
            current_solutions.append(path)
            return

        ## At most, explore 3 extra nodes before calling it quits
        if explore_count > 3:
            return

        new_block = sequence_index in sequence.contiguous_block_indices
        target = sequence.data[sequence_index]
        position = path[-1] if path else (0, 0)

        for it in range(breach_protocol_data.matrix_size)[:: 1 if in_row else -1]:
            current = (it, position[1]) if in_row else (position[0], it)
            if current not in path:
                ## Can progress to the next sequence index if target matches current
                progress = target == breach_protocol_data.data[current[1]][current[0]]

                ## Can't explore if in the middle of a contiguous block
                if not progress and not new_block:
                    continue

                _recursive_solve(
                    path + (current,),
                    not in_row,
                    sequence,
                    sequence_index
                    + int(progress),  ## forgive me father for i have sinned
                    explore_count + int(not progress),
                )

    shortest_solution_path = None
    solution_valid = False
    for potential_solution in sorted_converted_potential_solutions:
        current_solutions = list()
        _recursive_solve(tuple(), True, potential_solution, 0, 0)

        if current_solutions:
            solution_path = min(current_solutions, key=len)
            current_valid = len(solution_path) <= breach_protocol_data.buffer_size

            shorter_than_current_best = shortest_solution_path is None or len(
                solution_path
            ) < len(shortest_solution_path)
            if current_valid:
                if not solution_valid or shorter_than_current_best:
                    shortest_solution_path = solution_path
                    solution_valid = True
            elif not solution_valid and shorter_than_current_best:
                shortest_solution_path = solution_path

    ## Calculate true solution from the shortest solution path
    shortest_solution = None
    if shortest_solution_path:
        shortest_solution = tuple(
            breach_protocol_data.data[y][x] for x, y in shortest_solution_path
        )

    all_sequence_paths = tuple(it.data for it in sorted_converted_potential_solutions)
    sequence_path_data = models.SequencePathData(
        all_sequence_paths=all_sequence_paths,
        shortest_solution=shortest_solution,
        shortest_solution_path=shortest_solution_path,
        solution_valid=solution_valid,
    )
    LOG.debug(f"Sequence solver found sequence path data: {sequence_path_data}")
    return sequence_path_data


def autohack(
    config: models.Config,
    breach_protocol_data: models.BreachProtocolData,
    screenshot_data: models.ScreenshotData,
    sequence_path_data: models.SequencePathData,
):
    """Clicks the sequence solution in the game window."""
    _focus_game_window()
    in_row = True
    key_sequence: List[str] = list()
    cursor_position = (0, 0)
    key_names = (("left", "right"), ("up", "down"))
    shortest_solution_path: Tuple[Tuple[int, int], ...] = sequence_path_data.shortest_solution_path  # type: ignore
    for path_index, coordinate in enumerate(shortest_solution_path):
        dimension_index = 0 if in_row else 1
        delta = coordinate[dimension_index] - cursor_position[dimension_index]

        if delta != 0:
            key = key_names[dimension_index][delta > 0]
            check_start = -1 if delta < 0 else 1
            for it in range(check_start, delta + check_start, check_start):
                if in_row:
                    check = (cursor_position[0] + it, cursor_position[1])
                else:
                    check = (cursor_position[0], cursor_position[1] + it)
                if check not in shortest_solution_path[:path_index]:
                    key_sequence.append(key)
        in_row = not in_row
        cursor_position = coordinate
        key_sequence.append("f")
    LOG.debug(f"Autohacker built key sequence: {key_sequence}")
    pyautogui.press("right")
    time.sleep(0.05)
    pyautogui.press("left")
    time.sleep(0.05)
    for key in key_sequence:
        pyautogui.press(key)
