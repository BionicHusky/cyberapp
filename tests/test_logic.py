import os
import pathlib

from typing import List, Optional, Tuple, Type

import pytest  # type: ignore

import cpah


SCREENSHOTS_DIRECTORY = (
    pathlib.Path(os.path.realpath(__file__)).parent.resolve() / "screenshots"
)


class MatrixTestCase:
    def __init__(
        self,
        test_id: str,
        data: List[List[int]],
        buffer_size: int,
        sequences: List[List[int]],
        expected_valid: bool,
        expected_solution_length: Optional[int] = None,
        expected_solution_path: Optional[Tuple[Tuple[int, int], ...]] = None,
        expected_solvable: bool = True,
        selected_sequence_indices: Optional[Tuple[int, ...]] = None,
    ):
        self.test_id = test_id
        self.breach_protocol_data = cpah.models.BreachProtocolData(
            data=data,
            matrix_size=len(data),
            buffer_size=buffer_size,
            sequences=sequences,
            targets=[""] * len(sequences),
        )
        self.expected_valid = expected_valid
        self.expected_solution_length = expected_solution_length
        self.expected_solution_path = expected_solution_path
        self.expected_solvable = expected_solvable
        self.selected_sequence_indices = selected_sequence_indices

    def verify(self):
        sequence_path_data = cpah.logic.calculate_sequence_path_data(
            self.breach_protocol_data,
            selected_sequence_indices=self.selected_sequence_indices,
        )
        assert sequence_path_data.solution_valid == self.expected_valid
        if self.expected_solvable:
            assert sequence_path_data.shortest_solution is not None
            if self.expected_solution_length is not None:
                assert (
                    len(sequence_path_data.shortest_solution)
                    == self.expected_solution_length
                )
            assert sequence_path_data.shortest_solution_path is not None
            if self.expected_solution_path is not None:
                assert (
                    sequence_path_data.shortest_solution_path
                    == self.expected_solution_path
                )
        else:
            assert sequence_path_data.shortest_solution is None
            assert sequence_path_data.shortest_solution_path is None


matrix_test_data = [
    MatrixTestCase(
        test_id="Valid, 6 total nodes",
        data=[
            [0, 1, 0, 0, 0],
            [4, 1, 0, 3, 3],
            [3, 1, 3, 3, 4],
            [4, 1, 1, 1, 1],
            [4, 0, 1, 0, 0],
        ],
        buffer_size=6,
        sequences=[[0, 1], [1, 3], [3, 4, 0]],
        expected_valid=True,
        expected_solution_length=6,
    ),
    MatrixTestCase(
        test_id="Invalid, 6 total nodes",
        data=[
            [1, 0, 0, 1, 0],
            [0, 0, 0, 1, 3],
            [1, 3, 4, 1, 0],
            [3, 3, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        buffer_size=5,
        sequences=[[0, 3], [1, 0, 0], [0, 0, 1]],
        expected_valid=False,
        expected_solution_length=6,
    ),
    MatrixTestCase(
        test_id="Valid, 1 exploration node",
        data=[
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 4, 3],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        buffer_size=5,
        sequences=[[1, 2], [3, 4]],
        expected_valid=True,
        expected_solution_length=5,
        expected_solution_path=((1, 0), (1, 4), (5, 4), (5, 3), (4, 3)),
    ),
    MatrixTestCase(
        test_id="Valid, 3 exploration nodes",
        data=[
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 4],
            [0, 0, 0, 0, 0, 0],
        ],
        buffer_size=7,
        sequences=[[1, 2], [3, 4]],
        expected_valid=True,
        expected_solution_length=7,
    ),
    MatrixTestCase(
        test_id="Invalid, 3 exploration nodes",
        data=[
            [0, 4, 4, 1, 0, 3],
            [2, 0, 3, 0, 2, 3],
            [2, 0, 2, 0, 3, 2],
            [1, 2, 1, 0, 4, 1],
            [0, 4, 1, 2, 0, 3],
            [1, 1, 0, 0, 0, 1],
        ],
        buffer_size=6,
        sequences=[[0, 3, 4], [2, 2, 0], [2, 3, 0]],
        expected_valid=False,
        expected_solution_length=12,
    ),
    MatrixTestCase(
        test_id="Invalid, unsolvable",
        data=[
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 3, 2],
            [0, 0, 0, 4, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        buffer_size=20,
        sequences=[[1, 2], [3, 4]],
        expected_valid=False,
        expected_solvable=False,
    ),
    MatrixTestCase(
        test_id="Valid, sequence combination and splits",
        data=[
            [1, 0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 3, 4, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        buffer_size=8,
        sequences=[[1, 2, 2], [2, 3], [3, 4, 1]],
        expected_valid=True,
        expected_solution_length=8,
    ),
    MatrixTestCase(
        test_id="Valid, selected indices",
        data=[
            [3, 1, 0, 3, 2, 1],
            [1, 2, 0, 1, 3, 4],
            [4, 2, 1, 3, 0, 2],
            [1, 4, 3, 1, 2, 4],
            [0, 0, 1, 3, 3, 3],
            [0, 0, 0, 1, 4, 2],
        ],
        buffer_size=6,
        sequences=[[1, 4, 2], [0, 2, 4], [2, 4, 3, 1]],
        expected_valid=True,
        expected_solution_length=6,
        selected_sequence_indices=(0, 2),
    ),
    MatrixTestCase(
        test_id="Invalid, unsolvable from too many solutions",
        data=[
            [3, 3, 4, 1, 1, 0],
            [0, 4, 1, 0, 4, 0],
            [4, 1, 3, 1, 2, 1],
            [2, 4, 0, 0, 3, 1],
            [3, 2, 2, 2, 1, 3],
            [2, 2, 4, 3, 3, 2],
        ],
        buffer_size=6,
        sequences=[[1, 0, 0], [2, 1, 1, 1], [1, 2, 3], [1, 2, 2], [1, 3]],
        expected_valid=False,
        expected_solvable=False,
    ),
]


@pytest.mark.parametrize(
    "matrix_test_case", matrix_test_data, ids=[it.test_id for it in matrix_test_data]
)
def test_sequence_path_calculation(matrix_test_case):
    """
    Runs the sequence path calculation against
    the matrix data and checks the result.
    """
    matrix_test_case.verify()


class ScreenshotTestCase:
    def __init__(
        self,
        test_id: str,
        screenshot_name: str,
        matrix_data: Optional[List[List[int]]] = None,
        buffer_size: Optional[int] = None,
        sequences: Optional[List[List[int]]] = None,
        targets: Optional[List[str]] = None,
        raises: Optional[Type[Exception]] = None,
        **config_kwargs,
    ):
        self.test_id = test_id
        self.screenshot_name = screenshot_name
        self.matrix_data = matrix_data
        self.buffer_size = buffer_size
        self.sequences = sequences
        self.targets = targets
        self.raises = raises
        self.config = cpah.models.Config(**config_kwargs)

    def _verify(self):
        screenshot_data = cpah.logic.grab_screenshot(
            self.config, SCREENSHOTS_DIRECTORY / self.screenshot_name
        )
        screen_bounds = cpah.logic.parse_screen_bounds(self.config, screenshot_data)

        data = cpah.logic.parse_matrix_data(self.config, screenshot_data, screen_bounds)
        if self.matrix_data is not None:
            assert self.matrix_data == data

        buffer_size = cpah.logic.parse_buffer_size_data(
            self.config, screenshot_data, screen_bounds
        )
        if self.buffer_size is not None:
            assert self.buffer_size == buffer_size

        sequences = cpah.logic.parse_sequences_data(
            self.config, screenshot_data, screen_bounds
        )
        if self.sequences is not None:
            assert self.sequences == sequences

        targets = cpah.logic.parse_targets_data(
            self.config, screenshot_data, screen_bounds, len(sequences)
        )
        if self.targets is not None:
            assert self.targets == targets

    def verify(self):
        """
        Because pytest doesn't seem to support a conditional pytest.raises,
        this wraps the real _verify logic with pytest.raises as necessary.
        """
        if self.raises is None:
            self._verify()
        else:
            with pytest.raises(self.raises):
                self._verify()


screenshot_test_data = [
    ScreenshotTestCase(
        test_id="Standard size, basic",
        screenshot_name="0.png",
        matrix_data=[
            [0, 1, 0, 4, 0],
            [1, 1, 0, 0, 0],
            [1, 3, 0, 0, 4],
            [0, 0, 3, 0, 3],
            [0, 3, 4, 4, 3],
        ],
        buffer_size=6,
        sequences=[[0, 4], [0, 0], [0, 0, 4]],
        targets=["DATAMINE_V1", "DATAMINE_V2", "DATAMINE_V3"],
    ),
    ScreenshotTestCase(
        test_id="Standard size, prehacked",
        screenshot_name="1.png",
        matrix_data=[
            [3, 2, 5, 1, 1, 1, 3],
            [5, 5, 0, 0, 5, 1, 3],
            [5, 5, 1, 0, 1, 0, 1],
            [0, 3, 4, 3, 1, 0, 3],
            [1, 2, 0, 5, 2, 4, 1],
            [0, 4, 1, 3, 3, 5, 2],
            [0, 0, 2, 5, 1, 3, 5],
        ],
        buffer_size=8,
        sequences=[[5, 5], [0, 0, 5]],
        targets=["CAMERA SHUTDOWN", "MASS VULNERABILITY"],
    ),
    ScreenshotTestCase(
        test_id="Standard size, invalid cursor position",
        screenshot_name="2.png",
        raises=cpah.exceptions.CPAHMatrixParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid thresholds",
        screenshot_name="3.png",
        raises=cpah.exceptions.CPAHBufferParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, default thresholds, buffer size override",
        screenshot_name="3.png",
        matrix_data=[
            [1, 0, 3, 3, 3, 4],
            [0, 1, 0, 2, 1, 1],
            [0, 3, 1, 0, 2, 3],
            [1, 0, 0, 0, 0, 2],
            [0, 0, 2, 3, 4, 0],
            [1, 0, 2, 0, 2, 0],
        ],
        buffer_size=6,
        sequences=[[1, 0, 2], [0, 0, 0], [0, 3, 2]],
        targets=["DATAMINE_V1", "DATAMINE_V2", "DATAMINE_V3"],
        buffer_size_override=6,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid matrix threshold",
        screenshot_name="3.png",
        matrix_code_detection_threshold=0.1,
        raises=cpah.exceptions.CPAHMatrixParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Small size, invalid sequence threshold, buffer size override",
        screenshot_name="3.png",
        buffer_size_override=6,
        sequence_code_detection_threshold=0.1,
        raises=cpah.exceptions.CPAHSequenceParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Medium size, invalid buffer threshold",
        screenshot_name="4.png",
        raises=cpah.exceptions.CPAHBufferParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Medium size, valid buffer threshold",
        screenshot_name="4.png",
        buffer_box_detection_threshold=0.6,
        matrix_data=[
            [0, 0, 3, 0, 2, 1],
            [1, 1, 1, 0, 0, 4],
            [0, 2, 0, 0, 0, 2],
            [2, 1, 1, 1, 0, 0],
            [2, 0, 1, 4, 2, 2],
            [3, 0, 0, 0, 3, 3],
        ],
        buffer_size=6,
        sequences=[[1, 0, 2, 1]],
        targets=["ICEPICK"],
    ),
    ScreenshotTestCase(
        test_id="Standard size, several targets",
        screenshot_name="5.png",
        matrix_data=[
            [3, 3, 4, 1, 1, 0],
            [0, 4, 1, 0, 4, 0],
            [4, 1, 3, 1, 2, 1],
            [2, 4, 0, 0, 3, 1],
            [3, 2, 2, 2, 1, 3],
            [2, 2, 4, 3, 3, 2],
        ],
        buffer_size=6,
        sequences=[[1, 0, 0], [2, 1, 1, 1], [1, 2, 3], [1, 2, 2], [1, 3]],
        targets=[
            "ICEPICK",
            "FRIENDLY TURRETS",
            "MASS VULNERABILITY",
            "TURRET SHUTDOWN",
            "CAMERA SHUTDOWN",
        ],
    ),
    ScreenshotTestCase(
        test_id="Invalid screenshot",
        screenshot_name="6.png",
        raises=cpah.exceptions.CPAHScreenshotParseFailedException,
    ),
    ScreenshotTestCase(
        test_id="Ultrawide size",
        screenshot_name="7.png",
        matrix_data=[
            [0, 0, 1, 4, 1],
            [1, 3, 4, 0, 3],
            [0, 3, 1, 4, 4],
            [0, 3, 3, 0, 1],
            [1, 1, 0, 1, 1],
        ],
        buffer_size=8,
        sequences=[[1, 3], [3, 1], [1, 0, 1]],
        targets=["DATAMINE_V1", "DATAMINE_V2", "DATAMINE_V3"],
    ),
]


@pytest.mark.parametrize(
    "screenshot_test_case",
    screenshot_test_data,
    ids=[it.test_id for it in screenshot_test_data],
)
def test_screenshot_parsing(screenshot_test_case):
    """Runs the screenshot parsing logic and checks the result."""
    screenshot_test_case.verify()
