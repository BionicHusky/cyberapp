from typing import Optional

import system_hotkey  # type: ignore

from . import constants


class CPAHException(Exception):
    critical = False


class CPAHThreadRunningException(CPAHException):
    pass


class CPAHInvalidConfigException(CPAHException):
    critical = True

    def __init__(self, message: str):
        super().__init__(
            f"The configuration file at {constants.CONFIG_FILE_PATH} "
            f"is invalid or corrupt:\n\n{message}"
        )


class CPAHInvalidNewConfigException(CPAHException):
    def __init__(self, exception: ValueError):
        super().__init__(f"The configuration is not valid:\n\n{exception}")


class CPAHHotkeyRegistrationExceptions(CPAHException):
    def __init__(self, exception: system_hotkey.SystemRegisterError):
        super().__init__(
            "The analysis hotkey could not be registered. "
            f"Is another instance of the tool open?\n\nGiven error: {exception}"
        )


class CPAHGameNotFoundException(CPAHException):
    def __init__(self):
        super().__init__(f"Game window not found: {constants.GAME_EXECUTABLE_TITLE}")


class CPAHScreenshotParseFailedException(CPAHException):
    def __init__(self, message: str):
        super().__init__(
            f"{message}\n\nIs the breach protocol minigame screen active? "
            "If the confidence value is close to the threshold, you can lower the "
            "threshold of core elements detection in the configuration screen."
        )


class CPAHDataParseFailedException(CPAHException):
    def __init__(
        self,
        message: str,
        detection_type: str = "some",
        post: Optional[str] = None,
    ):
        combined_message = (
            f"{message}\n\nIf you are playing at a resolution smaller than "
            f"{constants.ANALYSIS_IMAGE_SIZE[0]}x{constants.ANALYSIS_IMAGE_SIZE[1]}, "
            f"you may need to decrease {detection_type} detection thresholds. "
            "Additionally, ensure your mouse cursor is not in the way of elements."
        )
        if post:
            combined_message += f"\n\n{post}"
        super().__init__(combined_message)


class CPAHBufferParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(
            message,
            detection_type="buffer box",
            post=(
                "If adjusting the detection thresholds does not fix the problem, "
                "you can override the buffer size to bypass automatic detection."
            ),
        )


class CPAHMatrixParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="matrix code")


class CPAHSequenceParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="sequence code")


class CPAHTargetParseFailedException(CPAHDataParseFailedException):
    def __init__(self, message: str):
        super().__init__(message, detection_type="target")
