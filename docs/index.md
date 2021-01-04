# CPAH: Cyberpunk 2077 Autohacker

This is a tool to help make the Cyberpunk 2077 Breach Protocol hacking minigame less tedious.
Check out the video below for a quick demonstration:

<video width="100%" controls muted autoplay loop>
  <source src="media/demo.mp4" type="video/mp4">
  Your browser does not support HTML5 Video.
</video>

## Features

* Automatic matrix code and sequence detection using OpenCV
* Selectable targets in case your buffer size is not large enough
* Autohacking by sending keys to solve the matrix for you
* Configurable hotkey to prevent the need to switch windows
* Sound notifications for people with only one monitor
* Configurable detection and override settings if you are playing at a small resolution

Internally, the code is linted with [Black](https://github.com/psf/black){target=_blank}
and [mypy](https://github.com/python/mypy){target=_blank},
tested with [pytest](https://github.com/pytest-dev/pytest){target=_blank},
frozen with [PyInstaller](https://github.com/pyinstaller/pyinstaller){target=_blank},
built with GitLab CI Pipelines, and hosted in the GitLab Package Registry.

I'm open to ideas! If you'd like to see something changed or if there's a bug,
feel free to create an issue on the project page.

## Download

[Releases can be found here.](https://gitlab.com/jkchen2/cpah/-/releases){target=_blank}
Download the `cpah.exe` package under the latest release:
![package](media/download_00_package.png)

!!! warning
    Windows and your web browser may tell you that this file is risky.
    You'll just have to take my word for it that it's not --
    you're welcome to check out the source code for yourself.

    Alternatively, if you're willing to give me money so that I can buy a code signing license,
    I'd be open to that :^)

If you're wondering why the binary is about 100 MB large, see the [FAQ](#faq).

## Usage

When you start up the tool, you will be greeted with this interface:
![interface](media/usage_00_interface.png)

The sections below detail what each button does.

### Analysis

To begin analysis, click the `Analyze` button. Alternatively, you can press the hotkey
if you have it configured (by default, this is ++ctrl+shift+h++).

CPAH will focus the Cyberpunk game window, take a screenshot, and look for key parts of the
breach protocol screen (such as the text, code matrix, sequences, and buffer size).
If these elements are found, CPAH will display these elements on the UI.

CPAH will try to find the shortest solution from all the given targets.
If a solution is found and your buffer size is large enough, it will display the solution
over the matrix, as well as the list of targets to the right:

!!! note
    Analysis works best if you are running your game at 1080p. If you are running at a lower
    resolution, you may need to lower some detection thresholds. See the
    [configuration section below](#detection-thresholds) for more.

![solution](media/usage_01_solution.png)

If a solution is found but it is too long, you can click on the targets on the right to choose
which targets you want to keep, and which ones to ignore:

![invalid_targets](media/usage_02_invalid_targets.png)
![valid_targets](media/usage_03_valid_targets.png)

If you have configured CPAH to enable automatic autohacking, autohacking will begin immediately
after analysis if a solution can be found for all targets.

### Autohacking

If CPAH found a valid solution, autohacking will become available. Clicking the `Autohack`
button will focus the Cyberpunk game window and press a combination of
++up++, ++down++, ++left++, ++right++, and ++f++ keys to automatically solve the minigame for you
based on the targets you selected.

!!! warning
    **It is very important that you do not move your mouse while CPAH is autohacking!**
    Moving the mouse will reset the position of the cursor in the code matrix, which will make
    CPAH input an invalid solution.

### Configuration

CPAH can be configured to be more friendly for single monitor users, and for those running
Cyberpunk at a resolution smaller than 1080p.

![configuration_interface](media/usage_04_configuration_interface.png)
![configuration_detection](media/usage_05_configuration_detection.png)

Each section in the configuration screen has a small section detailing what it does, which
should provide enough information on what it does. However, here are a few more details
for some specific options:

#### Analysis hotkey

The analysis hotkey field defines the hotkey that runs analysis if pressed. It is a `+` delimited
list of keys. By default, this sequence is `control + shift + h`. Below is a table listing some
example hotkeys:

| Hotkey sequence        | Keys                 |
| ---------------------- | -------------------- |
| `control + shift + h`  | ++control+shift+h++  |
| `super + f1`           | ++windows+f1++       |
| `control + alt + kp_5` | ++control+alt+num5++ |
| `control + home`       | ++control+home++     |

#### Detection thresholds

If you are playing at a resolution smaller than 1080p (1920 x 1080), CPAH may fail to correctly
read screen elements. You can configure the thresholds for the detection of certain elements if
you find that CPAH is having issues.

Here is a screenshot with elements labeled:

![detection_legend](media/usage_06_detection_legend.png)

| Color  | Description           | Detection default |
| ------ | --------------------- | ----------------- |
| Red    | Core text elements    | 0.8               |
| Orange | Matrix codes          | 0.8               |
| Cyan   | Buffer boxes          | 0.7               |
| Blue   | Target sequence codes | 0.7               |
| Purple | Target names          | 0.8               |

The defaults are values that work well if playing the game at 1080p.
The lower the resolution, the lower the detection threshold for certain elements need to be.
There isn't an exact mapping between screen resolutions and detection thresholds,
so you may need to play around with them and see what works for you.

!!! note
    Buffer box detection is the hardest at lower resolutions -- sometimes lowering the detection
    for buffer boxes doesn't help. In this case, you can set a buffer size override instead.

## Debugging

If CPAH encounters an unhandled error, it will display a traceback and exit.
All logs are also recorded to a log file located at
`%AppData%\cp2077_autohack\log.txt`.
If sending information in to debug a problem, this is an important file to include.

Additionally, if there is a problem with the configuration file being corrupt,
it can be manually edited or removed at
`%AppData%\cp2077_autohack\config.json`

## FAQ

**Q: Why is the binary so large?**

A: The binary bundles several Python modules with PyInstaller.
The biggest module is `opencv-python-headless`, which by itself accounts for about 50 MB.
The rest is a mix of `PySide2` (the Qt framework), `Pillow`, `pydantic`, and a few others.
All dependencies are bundled together as to avoid requiring the user to install anything else.

**Q: Why does it take so long to open?**

A: As a continuation of the previous answer, PyInstaller freezes the code into a single executable.
This executable is effectively a zip containing all of the code and an entire Python runtime.
Each time CPAH is launched, your computer has to extract the data to a temporary directory before
it can actually run the tool.

**Q: What if I don't want CPAH to take a screenshot?**

A: I promise there's nothing malicious going on -- the screenshot data is localized to the game
window and is never sent anywhere (i.e. it is processed locally on your machine). However, if
you truly don't want to have CPAH take a screenshot, you can save a screenshot you take yourself
of the breach protocol screen (fullscreen works best), then right click the `Analyze` button
and select the screenshot.
