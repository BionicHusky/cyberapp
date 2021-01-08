import dataclasses
import functools
import os
import pathlib
import sys
import types
import winsound

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import pyautogui  # type: ignore
import system_hotkey  # type: ignore

from PIL import Image  # type: ignore
from PySide2.QtCore import QFile, QObject, QRect, Qt, QThread, Signal
from PySide2.QtGui import QCloseEvent, QFontDatabase, QIcon, QPixmap
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import constants
from . import exceptions
from . import logic
from . import models

from .logger import LOG


## Workaround for certain unhandled exceptions using PyQt5 instead of PySide2
## And also ImageQt for using PyQt5
## SO: 27340162
import PySide2

sys.modules["pyQt5"] = PySide2
from PIL import ImageQt


class Worker(QObject):

    started = Signal()
    finished = Signal()
    error = Signal(models.Error)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._args = tuple()
        self._kwargs = dict()

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def wrapped_run(self):
        """Long-running task."""
        LOG.debug(f"Worker thread started: {type(self).__name__}")
        self.started.emit()
        try:
            self.run(*self._args, **self._kwargs)
        except Exception as exception:
            LOG.exception("Worker thread encountered an exception:")
            self.error.emit(models.Error(exception))
        LOG.debug(f"Worker thread finished: {type(self).__name__}")
        self.finished.emit()


@dataclasses.dataclass
class ThreadContainer:
    thread: QThread
    worker: Worker


def _beep_sequence(config: models.Config, frequencies: Iterable[int]):
    """Helper function for generating and playing notification beeps."""
    if config.enable_beeps:
        for frequency in frequencies:
            winsound.Beep(frequency, constants.BEEP_DURATION)


class SolverWorker(Worker):

    matrix_overlay_image_signal = Signal(Image.Image)
    buffer_overlay_image_signal = Signal(Image.Image)
    solution_signal = Signal(models.SequencePathData)
    sequence_path_data_available_signal = Signal(models.SequencePathData)

    def run(  # type: ignore
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        selected_sequence_indices: Tuple[int],
    ):
        self._solve_and_signal(
            config,
            breach_protocol_data,
            selected_sequence_indices=selected_sequence_indices,
            beep=False,
        )

    def _solve_and_signal(
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        selected_sequence_indices: Optional[Tuple[int]] = None,
        beep=True,
    ) -> models.SequencePathData:
        """
        Convenience function for solving the matrix,
        generating the images, and emitting the proper signals on the worker.
        """
        sequence_path_data = logic.calculate_sequence_path_data(
            breach_protocol_data, selected_sequence_indices=selected_sequence_indices
        )
        matrix_overlay_image = None
        if sequence_path_data.shortest_solution_path:
            image_size = (
                constants.MATRIX_IMAGE_SIZE * breach_protocol_data.matrix_size
                + constants.MATRIX_IMAGE_SPACING
                * (breach_protocol_data.matrix_size - 1)
            )
            matrix_overlay_image = logic.generate_sequence_path_image(
                image_size,
                sequence_path_data.shortest_solution_path,
                sequence_path_data.solution_valid,
            )
            self.matrix_overlay_image_signal.emit(matrix_overlay_image)  # type: ignore
        if sequence_path_data.solution_valid:
            buffer_sequence_image = logic.generate_buffer_sequence_image(
                sequence_path_data.shortest_solution  # type: ignore
            )
            self.buffer_overlay_image_signal.emit(buffer_sequence_image)  # type: ignore
            self.sequence_path_data_available_signal.emit(sequence_path_data)  # type: ignore
        self.solution_signal.emit(sequence_path_data)  # type: ignore
        if beep:
            beep_tone = (
                constants.BEEP_SUCCESS
                if sequence_path_data.solution_valid
                else constants.BEEP_FAIL
            )
            _beep_sequence(config, [constants.BEEP_START, beep_tone])
        return sequence_path_data


class AnalysisWorker(SolverWorker):
    status = Signal(str)
    matrix_image_signal = Signal(Image.Image)
    buffer_image_signal = Signal(Image.Image)
    sequences_signal = Signal(tuple)
    analysis_data_available_signal = Signal(models.BreachProtocolData)
    run_autohack_signal = Signal()
    analysis_finished_signal = Signal(bool)

    def run(  # type: ignore
        self, config: models.Config, from_file: Optional[pathlib.Path] = None
    ):
        self.status.emit("SCREEN")  # type: ignore
        screenshot_data = logic.grab_screenshot(config, from_file=from_file)

        self.status.emit("DATA POSITIONS")  # type: ignore
        screen_bounds = logic.parse_screen_bounds(config, screenshot_data)

        self.status.emit("CODE MATRIX")  # type: ignore
        data = logic.parse_matrix_data(config, screenshot_data, screen_bounds)
        image_size = constants.MATRIX_IMAGE_SIZE * len(
            data
        ) + constants.MATRIX_IMAGE_SPACING * (len(data) - 1)
        matrix_image = logic.generate_matrix_image(image_size, data)
        self.matrix_image_signal.emit(matrix_image)  # type: ignore

        self.status.emit("BUFFER SIZE")  # type: ignore
        buffer_size = logic.parse_buffer_size_data(
            config, screenshot_data, screen_bounds
        )
        buffer_boxes_image = logic.generate_buffer_boxes_image(buffer_size)
        self.buffer_image_signal.emit(buffer_boxes_image)  # type: ignore

        self.status.emit("SEQUENCES")  # type: ignore
        sequences = logic.parse_sequences_data(config, screenshot_data, screen_bounds)

        self.status.emit("TARGETS")  # type: ignore
        targets = logic.parse_targets_data(
            config, screenshot_data, screen_bounds, len(sequences)
        )
        self.sequences_signal.emit(tuple(zip(sequences, targets)))  # type: ignore
        breach_protocol_data = models.BreachProtocolData(
            data=data,
            matrix_size=len(data),
            buffer_size=buffer_size,
            sequences=sequences,
            targets=targets,
        )
        analysis_data = models.AnalysisData(
            breach_protocol_data=breach_protocol_data,
            screenshot_data=screenshot_data,
            screen_bounds=screen_bounds,
        )
        self.analysis_data_available_signal.emit(analysis_data)  # type: ignore

        self.status.emit("SOLUTIONS")  # type: ignore
        sequence_path_data = self._solve_and_signal(config, breach_protocol_data)

        run_autohack = (
            sequence_path_data.solution_valid and config.auto_autohack and not from_file
        )
        if run_autohack:
            self.run_autohack_signal.emit()  # type: ignore
        self.analysis_finished_signal.emit(run_autohack)  # type: ignore


class AutohackWorker(Worker):
    def run(  # type: ignore
        self,
        config: models.Config,
        breach_protocol_data: models.BreachProtocolData,
        screenshot_data: models.ScreenshotData,
        sequence_path_data: models.SequencePathData,
    ):
        _beep_sequence(config, [constants.BEEP_START])
        logic.autohack(
            config, breach_protocol_data, screenshot_data, sequence_path_data
        )
        _beep_sequence(config, [constants.BEEP_SUCCESS])


class ErrorHandlerMixin:
    """Mixin to add exception handling."""

    _error_handler_signal = Signal(models.Error, QWidget)

    def __init__(self, *args, **kwargs):
        super(ErrorHandlerMixin, self).__init__(*args, **kwargs)
        self.error_box = QMessageBox()
        self._error_handler_signal.connect(self.show_error_message)

    def _handle_exceptions(method):
        """Adds exception handling to the method."""
        self_focus = getattr(method, "_EHM_self_focus", True)

        @functools.wraps(method)
        def _decorated(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as exception:
                self._error_handler_signal.emit(
                    models.Error(exception), self if self_focus else None
                )

        _decorated._EHM_nonce = True
        return _decorated

    @staticmethod
    def class_decorator(cls):
        """Decorates all functions of the widget to have error handling"""
        for attr in cls.__dict__:
            method = getattr(cls, attr)
            if (
                isinstance(method, types.FunctionType)
                and not getattr(method, "_EHM_nonce", False)
                and not getattr(method, "_EHM_ignore", False)
            ):
                setattr(cls, attr, ErrorHandlerMixin._handle_exceptions(method))
        return cls

    @staticmethod
    def ignore(method):
        """Decorator modifier to ignore error handling by the mixin."""
        method._EHM_ignore = True
        return method

    @staticmethod
    def no_self_focus(method):
        """Decorator modifier to disable self-focus on error."""
        method._EHM_self_focus = False
        return method

    def show_error_message(self, error: models.Error, focus: Optional[QWidget] = None):
        LOG.error(f"Displaying error message: {error}")
        if error.critical:
            button, dialog = QMessageBox.Close, self.error_box.critical
        else:
            button, dialog = QMessageBox.Ok, self.error_box.warning
        dialog(focus, error.title, error.message, button)
        if error.critical:
            LOG.info("Closing application due to critical error")
            ## Attempting to exit with QApplication.exit causes a stack overflow. Cool.
            sys.exit(2)


@ErrorHandlerMixin.class_decorator
class CPAH(ErrorHandlerMixin, QWidget):

    analysis_hotkey_signal = Signal()

    def __init__(self):
        super().__init__()

        ## Internal
        self.autohack_warning_box = QMessageBox()
        self.screenshot_picker_dialog = QFileDialog()
        self._threads: Dict[str, ThreadContainer] = dict()
        self.hotkey = system_hotkey.SystemHotkey()

        def _error_handle_wrap_hotkey(method):
            def _decorated(*args, **kwargs):
                try:
                    return method(*args, **kwargs)
                except system_hotkey.SystemRegisterError as exception:
                    self._error_handler_signal.emit(
                        models.Error(
                            exceptions.CPAHHotkeyRegistrationExceptions(exception)
                        )
                    )

            return _decorated

        ## Patch system_hotkey so that errors are properly caught when using Qt
        self.hotkey._the_grab = _error_handle_wrap_hotkey(self.hotkey._the_grab)

        ## State
        self.analyzing = False
        self.autohacking = False

        ## Cache
        self.analysis_data: models.AnalysisData = None
        self.sequence_path_data: models.SequencePathData = None
        self.buffer_image_cache: Image.Image = None

        self.bootstrap()
        self.load_ui()

    @ErrorHandlerMixin.ignore
    def _set_up_worker(
        self, name: str, worker: Type[Worker]
    ) -> Tuple[Callable, ThreadContainer]:
        """Convenience function for setting up a worker thread."""
        thread_container = self._threads.get(name)
        if thread_container:
            try:
                if thread_container.thread.isRunning():
                    raise exceptions.CPAHThreadRunningException
            except RuntimeError:  ## Thread already deleted
                pass

        self._threads[name] = thread_container = ThreadContainer(QThread(), worker())
        thread_container.worker.moveToThread(thread_container.thread)
        thread_container.thread.started.connect(thread_container.worker.wrapped_run)
        thread_container.thread.finished.connect(thread_container.thread.deleteLater)
        thread_container.worker.finished.connect(thread_container.thread.quit)  # type: ignore
        thread_container.worker.finished.connect(thread_container.worker.deleteLater)  # type: ignore
        thread_container.worker.error.connect(self.show_error_message)  # type: ignore

        def _start_worker(*args, **kwargs):
            thread_container.worker._args = args
            thread_container.worker._kwargs = kwargs
            thread_container.thread.start()

        return _start_worker, thread_container

    @ErrorHandlerMixin.ignore
    def show_error_message(self, error: models.Error, focus: Optional[QWidget] = None):
        super().show_error_message(error, focus=focus or self)
        LOG.info("Resetting displays due to error")
        self.reset_displays()
        self.analyzing = False
        self.autohacking = False

    def cache_analysis_data(self, data: models.AnalysisData):
        LOG.debug(f"Caching analysis data: {data}")
        self.analysis_data = data

    def cache_sequence_path_data(self, data: models.SequencePathData):
        LOG.debug(f"Caching sequence path data: {data}")
        self.sequence_path_data = data

    def start_screenshot_analysis(self):
        """
        Like start_analysis, but uses an existing
        screenshot instead of taking a new one.
        """
        LOG.debug("start_screenshot_analysis")
        self.screenshot_picker_dialog.setFileMode(QFileDialog.ExistingFile)
        self.screenshot_picker_dialog.setNameFilter("Images (*.png *.jpg)")
        if self.screenshot_picker_dialog.exec_():
            selected_file = self.screenshot_picker_dialog.selectedFiles()[0]
            LOG.debug(f"User selected screenshot: {selected_file}")
            self.start_analysis(from_file=pathlib.Path(selected_file))

    def start_analysis(self, from_file: Optional[pathlib.Path] = None):
        LOG.debug("start_analysis")
        if self.autohacking:
            LOG.debug("Ignoring analysis start because autohacking is in progress.")
            return
        elif self.analyzing:
            LOG.debug("Ignoring analysis start because analysis is still running.")
            return
        try:
            start_worker, thread_container = self._set_up_worker(
                "analysis", AnalysisWorker
            )
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Analysis worker already running! Ignoring start...")
            return

        self.analyzing = True

        ## Clear all images and cache
        self.reset_displays()
        self.analysis_data = None  # type: ignore
        self.sequence_path_data = None  # type: ignore

        ## Disable the analyze button until it's done
        self.analyze_button.setEnabled(False)

        _beep_sequence(self.config, [constants.BEEP_START])

        thread_container.worker.matrix_image_signal.connect(self.set_matrix_image)
        thread_container.worker.matrix_overlay_image_signal.connect(
            self.set_matrix_overlay_image
        )
        thread_container.worker.buffer_image_signal.connect(self.set_buffer_image)
        thread_container.worker.buffer_overlay_image_signal.connect(
            self.set_buffer_overlay_image
        )
        thread_container.worker.status.connect(self.show_analyzing_status)
        thread_container.worker.analysis_finished_signal.connect(self.analysis_finished)
        thread_container.worker.sequences_signal.connect(
            self.sequence_container.set_sequences
        )
        thread_container.worker.solution_signal.connect(self.show_solution)
        thread_container.worker.analysis_data_available_signal.connect(
            self.cache_analysis_data
        )
        thread_container.worker.sequence_path_data_available_signal.connect(
            self.cache_sequence_path_data
        )
        thread_container.worker.run_autohack_signal.connect(self.start_autohack)
        start_worker(self.config, from_file=from_file)

    def analysis_finished(self, autohacking: bool = False):
        LOG.debug("analysis_finished")
        self.analyzing = False
        self.analyze_button.setEnabled(not autohacking)
        self._reset_analyze_button_text()

    def show_analyzing_status(self, status: str):
        LOG.debug(f"Analyzing stutus set to: {status}")
        self.analyze_button.setText(f"ANALYZING... [{status}]")

    def start_autohack(self):
        LOG.debug("start_autohack")

        if self.config.show_autohack_warning_message:
            LOG.debug("Showing first time autohack warning")
            self.autohack_warning_box.warning(
                self,
                "Autohack warning",
                (
                    "When running the autohack, it is important that you do not move "
                    "the mouse until the autohack is finished, because that will "
                    "reset the code matrix cursor position and will cause the "
                    "autohack to fail!\n\n"
                    "This message will only be displayed once."
                ),
                QMessageBox.Ok,
            )
            self.config.show_autohack_warning_message = False
            logic.save_config(self.config)

        try:
            start_worker, thread_container = self._set_up_worker(
                "autohack", AutohackWorker
            )
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Autohacker worker already running! Ignoring autohack...")
            return

        self.autohacking = True
        self.analyze_button.setEnabled(False)
        self.autohack_button.setEnabled(False)
        self.autohack_button.setText(f"AUTOHACKING...")
        thread_container.worker.finished.connect(self.autohack_finished)
        start_worker(
            self.config,
            self.analysis_data.breach_protocol_data,
            self.analysis_data.screenshot_data,
            self.sequence_path_data,
        )

    def autohack_finished(self):
        LOG.debug("autohack_finished")
        self.autohacking = False
        self.analyze_button.setEnabled(True)
        self.autohack_button.setEnabled(True)
        self.autohack_button.setText(f"AUTOHACK")

    def open_configuration(self):
        LOG.debug("open_configuration")
        self.configuration_screen.show()

    def configuration_changed(self, reset_displays: bool = True):
        LOG.debug("Resetting displays due to configuration")

        pyautogui.PAUSE = self.config.autohack_keypress_delay / 1000
        LOG.debug(f"Set pyautogui.PAUSE value to {pyautogui.PAUSE}")

        existing_binds = tuple(self.hotkey.keybinds)
        for existing_bind in existing_binds:
            LOG.debug(f"Unregistering bind {existing_bind}")
            self.hotkey.unregister(existing_bind)
        if self.config.analysis_hotkey:
            LOG.debug(f"Registering bind {self.config.analysis_hotkey}")
            self.hotkey.register(
                self.config.analysis_hotkey,
                callback=lambda _: self.analysis_hotkey_signal.emit(),  # type: ignore
            )

        if reset_displays:
            self.reset_displays()

    def show_solution(self, sequence_path_data: models.SequencePathData):
        LOG.debug("show_solution")
        if sequence_path_data.solution_valid:
            self.autohack_button.setEnabled(True)
        else:
            self.autohack_button.setEnabled(False)
            if sequence_path_data.shortest_solution:
                converted_solution = "  ".join(
                    logic.convert_code(sequence_path_data.shortest_solution)
                )
                self.show_buffer_error_message(
                    length=len(sequence_path_data.shortest_solution),
                    tooltip=converted_solution,
                    keep_path_overlay=True,
                )
            else:
                self.show_buffer_error_message()

    def recalculate_solution(self):
        LOG.debug("recalculate_solution")
        try:
            start_worker, thread_container = self._set_up_worker("solver", SolverWorker)
        except exceptions.CPAHThreadRunningException:
            LOG.warning("Solver worker already running! Ignoring recalculation...")
            return

        thread_container.worker.solution_signal.connect(self.show_solution)
        thread_container.worker.matrix_overlay_image_signal.connect(
            self.set_matrix_overlay_image
        )
        thread_container.worker.buffer_overlay_image_signal.connect(
            self.set_buffer_overlay_image
        )
        thread_container.worker.sequence_path_data_available_signal.connect(
            self.cache_sequence_path_data
        )

        enabled_selections = list()
        for index, sequence_selection in enumerate(self.sequence_container.sequences):
            if sequence_selection.selected:
                enabled_selections.append(index)

        ## All entries deselected
        if not enabled_selections:
            self.set_matrix_overlay_image()
            self.set_buffer_overlay_image()
        else:
            start_worker(
                self.config,
                self.analysis_data.breach_protocol_data,
                enabled_selections,
            )

    def _convert_and_set_image(self, image: Image.Image, label: QLabel):
        """Helper function to convert an image and set it on the label."""
        qt_image = ImageQt.ImageQt(image)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)

    def set_matrix_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting matrix image")
        if image is None:
            self.matrix_label.setPixmap(None)
            self.set_matrix_overlay_image()
            self.matrix_overlay_label.setStyleSheet(constants.LABEL_HINT_STYLE)
            self.matrix_overlay_label.setText("CODE MATRIX")
        else:
            self.matrix_overlay_label.setText("")
            self._convert_and_set_image(image, self.matrix_label)

    def set_matrix_overlay_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting matrix overlay image")
        if image is None:
            self.matrix_overlay_label.setPixmap(None)
        else:
            self._convert_and_set_image(image, self.matrix_overlay_label)

    def set_buffer_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting buffer image")
        if image is None:
            self.buffer_label.setPixmap(None)
            self.set_buffer_overlay_image()
            self.buffer_overlay_label.setStyleSheet(constants.LABEL_HINT_STYLE)
            self.buffer_overlay_label.setText("BUFFER")
        else:
            self.buffer_image_cache = image
            self.buffer_overlay_label.setText("")
            self._convert_and_set_image(image, self.buffer_label)

    def set_buffer_overlay_image(self, image: Optional[Image.Image] = None):
        LOG.debug("Setting buffer overlay image")
        if image is None:
            self.buffer_overlay_label.setPixmap(None)
        else:
            self.set_buffer_image(self.buffer_image_cache)
            self._convert_and_set_image(image, self.buffer_overlay_label)

    def show_buffer_error_message(
        self,
        length: Optional[int] = None,
        tooltip: Optional[str] = None,
        keep_path_overlay: bool = False,
    ):
        LOG.debug("Setting buffer error message")
        if not keep_path_overlay:
            self.set_matrix_overlay_image()
        self.set_buffer_image()
        self.buffer_overlay_label.setStyleSheet(constants.BUFFER_ERROR_STYLE)
        if length is None:
            message = "NO VALID PATH FOUND"
        else:
            message = f"SEQUENCE TOO LONG ({length})"
        self.buffer_overlay_label.setText(message)
        self.buffer_overlay_label.setToolTip(tooltip)

    def _reset_analyze_button_text(self):
        """Helper function to set the analyze button text."""
        text = "ANALYZE"
        if self.config.auto_autohack:
            text += " + AUTOHACK"
        self.analyze_button.setText(text)

    def reset_displays(self):
        """Resets the images/messages."""
        LOG.debug("Resetting displays")
        self._reset_analyze_button_text()
        self.analyze_button.setEnabled(True)
        self.autohack_button.setEnabled(False)
        self.set_matrix_image()
        self.set_buffer_image()
        self.sequence_container.set_sequences(None)
        self.analysis_data = None
        self.sequence_path_data = None
        self.buffer_image_cache = None

    def bootstrap(self):
        """Loads the config file and sets the system-wide autohack hotkey."""
        LOG.debug("Bootstrapping application")
        self.config = logic.load_config()
        self.analysis_hotkey_signal.connect(self.start_analysis)
        self.configuration_changed(reset_displays=False)

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile(str(constants.RESOURCES_DIRECTORY / "uis/main.ui"))
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

        ## Window properties
        self.setWindowIcon(QIcon(str(constants.IMAGES_DIRECTORY / "icon.ico")))
        self.setWindowTitle(f"{constants.APPLICATION_NAME} - {constants.VERSION}")
        self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)

        ## Font loading
        QFontDatabase.addApplicationFont(str(constants.FONT_PATH))

        ## Connect buttons
        self.analyze_button = self.findChild(QPushButton, "analyzeButton")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.analyze_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.analyze_button.customContextMenuRequested.connect(
            lambda _: self.start_screenshot_analysis()
        )
        self.autohack_button = self.findChild(QPushButton, "autohackButton")
        self.autohack_button.clicked.connect(self.start_autohack)
        self.configure_button = self.findChild(QPushButton, "configureButton")
        self.configure_button.clicked.connect(self.open_configuration)

        ## Labels to be referenced later for displaying images
        self.matrix_label = self.findChild(QLabel, "matrixLabel")
        self.matrix_overlay_label = self.findChild(QLabel, "matrixOverlayLabel")
        self.buffer_label = self.findChild(QLabel, "bufferLabel")
        self.buffer_overlay_label = self.findChild(QLabel, "bufferOverlayLabel")
        self.sequences_label = self.findChild(QLabel, "sequencesLabel")

        self.matrix_overlay_label.setAttribute(Qt.WA_TranslucentBackground, True)

        ## Load SequenceContainer into the scroll area
        self.sequence_container = SequenceContainer(self)
        self.scroll_area_widget = self.findChild(QScrollArea, "sequenceScrollArea")
        self.scroll_area_widget.setWidget(self.sequence_container)

        ## Load configuration screen
        self.configuration_screen = ConfigurationScreen(self)
        self.configuration_screen.configuration_changed_signal.connect(
            self.configuration_changed
        )

        self.reset_displays()


@ErrorHandlerMixin.class_decorator
class SequenceContainer(ErrorHandlerMixin, QWidget):
    """Container that holds all available SequenceSelections."""

    @ErrorHandlerMixin.no_self_focus
    def __init__(self, parent_widget: CPAH):
        super().__init__()
        self.parent_widget = parent_widget
        self.sequences: List[SequenceSelection] = list()
        layout = QVBoxLayout()
        self.setLayout(layout)

    @ErrorHandlerMixin.no_self_focus
    def set_sequences(self, sequences: Optional[List[Tuple[List[int], str]]]):
        layout = self.layout()
        for old_sequence in self.sequences:
            layout.removeWidget(old_sequence)
            old_sequence.deleteLater()

        self.sequences = list()
        for sequence, target in sequences or list():
            selection = SequenceSelection(sequence, target)
            selection.selection_updated.connect(self.parent_widget.recalculate_solution)
            layout.addWidget(selection)
            self.sequences.append(selection)
        calculated_height = 76 * len(self.sequences) - (
            12 * max(len(self.sequences) - 1, 0)
        )
        LOG.debug(f"Calculated sequence display height: {calculated_height}")
        self.setGeometry(QRect(0, 0, 367, calculated_height))
        self.parent_widget.sequences_label.setText(
            "" if self.sequences else "SEQUENCES"
        )


@ErrorHandlerMixin.class_decorator
class SequenceSelection(ErrorHandlerMixin, QFrame):
    """A sequence option that shows the sequence and reward type."""

    selection_updated = Signal()

    @ErrorHandlerMixin.no_self_focus
    def __init__(self, sequence: List[int], target: str):
        super().__init__()
        self.selected = True
        self.setCursor(Qt.PointingHandCursor)
        sequence_text = "  ".join(logic.convert_code(sequence))
        self.sequence_label = QLabel(self)
        self.sequence_label.setText(sequence_text)
        self.sequence_label.setGeometry(QRect(10, 30, 350, 20))
        self.target_label = QLabel(self)
        self.target_label.setText(target)
        self.target_label.setGeometry(QRect(10, 10, 350, 20))

    @ErrorHandlerMixin.no_self_focus
    def mouseReleaseEvent(self, event):
        self.selected = not self.selected
        color = "white" if self.selected else constants.SELECTION_OFF_COLOR
        self.setStyleSheet(f"color: {color};\nborder-color: {color};")
        self.selection_updated.emit()


@ErrorHandlerMixin.class_decorator
class ConfigurationScreen(ErrorHandlerMixin, QWidget):

    configuration_changed_signal = Signal()

    def __init__(self, parent_widget: CPAH):
        super().__init__()
        self.parent_widget = parent_widget
        self.load_ui()

    def load_ui(self):
        loader = QUiLoader()
        ui_file = QFile(str(constants.RESOURCES_DIRECTORY / "uis/settings.ui"))
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

        ## Window properties
        self.setWindowTitle("Configuration")
        self.setWindowFlags(self.windowFlags() | Qt.MSWindowsFixedSizeDialogHint)
        self.setWindowModality(Qt.ApplicationModal)

        ## Get inputs and set from configuration values
        self.automatic_autohacking_check_box = self.findChild(
            QCheckBox, "automaticAutohackingCheckBox"
        )
        self.beep_notifications_check_box = self.findChild(
            QCheckBox, "beepNotificationsCheckBox"
        )
        self.analysis_hotkey_line_edit = self.findChild(
            QLineEdit, "analysisHotkeyLineEdit"
        )
        self.autohack_keypress_delay_spin_box = self.findChild(
            QSpinBox, "autohackKeypressDelaySpinBox"
        )
        self.buffer_size_override_spin_box = self.findChild(
            QSpinBox, "bufferSizeOverrideSpinBox"
        )
        self.core_text_spin_box = self.findChild(QDoubleSpinBox, "coreTextSpinBox")
        self.matrix_codes_spin_box = self.findChild(
            QDoubleSpinBox, "matrixCodesSpinBox"
        )
        self.target_sequence_spin_box = self.findChild(
            QDoubleSpinBox, "targetSequenceSpinBox"
        )
        self.buffer_boxes_spin_box = self.findChild(
            QDoubleSpinBox, "bufferBoxesSpinBox"
        )
        self.target_names_spin_box = self.findChild(
            QDoubleSpinBox, "targetNamesSpinBox"
        )

        self.fill_values()

        ## Connect buttons
        self.ok_button = self.findChild(QPushButton, "okButton")
        self.ok_button.clicked.connect(lambda: self.close(save=True))
        self.cancel_button = self.findChild(QPushButton, "cancelButton")
        self.cancel_button.clicked.connect(self.close)

    def show(self):
        self.fill_values()
        super().show()

    def fill_values(self):
        """Fills in the configuration fields from loaded config values."""
        config = self.parent_widget.config
        self.automatic_autohacking_check_box.setChecked(config.auto_autohack)
        self.beep_notifications_check_box.setChecked(config.enable_beeps)
        self.analysis_hotkey_line_edit.setText(" + ".join(config.analysis_hotkey))
        self.autohack_keypress_delay_spin_box.setValue(config.autohack_keypress_delay)
        self.buffer_size_override_spin_box.setValue(config.buffer_size_override)
        self.core_text_spin_box.setValue(config.core_detection_threshold)
        self.matrix_codes_spin_box.setValue(config.matrix_code_detection_threshold)
        self.target_sequence_spin_box.setValue(config.sequence_code_detection_threshold)
        self.buffer_boxes_spin_box.setValue(config.buffer_box_detection_threshold)
        self.target_names_spin_box.setValue(config.target_detection_threshold)

    def closeEvent(self, event: QCloseEvent):
        self.close()

    def close(self, save: bool = False):
        LOG.debug(f"Closing configuration window (save={save})")
        if save:
            test = self.parent_widget.config.copy(deep=True)
            test.auto_autohack = self.automatic_autohacking_check_box.isChecked()
            test.enable_beeps = self.beep_notifications_check_box.isChecked()
            test.autohack_keypress_delay = self.autohack_keypress_delay_spin_box.value()
            test.buffer_size_override = self.buffer_size_override_spin_box.value()
            test.core_detection_threshold = self.core_text_spin_box.value()
            test.matrix_code_detection_threshold = self.matrix_codes_spin_box.value()
            test.sequence_code_detection_threshold = (
                self.target_sequence_spin_box.value()
            )
            test.buffer_box_detection_threshold = self.buffer_boxes_spin_box.value()
            test.target_detection_threshold = self.target_names_spin_box.value()
            hotkey_string = self.analysis_hotkey_line_edit.text().lower().strip()
            if hotkey_string:
                hotkeys = list(it.strip() for it in hotkey_string.split("+"))
            else:
                hotkeys = list()
            test.analysis_hotkey = hotkeys
            try:
                test.validate(test.dict())
            except ValueError as exception:
                raise exceptions.CPAHInvalidNewConfigException(exception)
            self.parent_widget.config = test
            logic.save_config(self.parent_widget.config)
            self.configuration_changed_signal.emit()  # type: ignore
        self.hide()


def start():
    LOG.info(f"Starting {constants.APPLICATION_NAME} {constants.VERSION}")
    constants.QAPPLICATION_INSTANCE.setAttribute(Qt.AA_EnableHighDpiScaling)
    try:
        widget = CPAH()
    except Exception as exception:
        LOG.exception("Application startup exception:")
        error = models.Error(exception)
        QMessageBox().critical(None, error.title, error.message, QMessageBox.Close)
        sys.exit(1)
    widget.show()
    sys.exit(constants.QAPPLICATION_INSTANCE.exec_())
