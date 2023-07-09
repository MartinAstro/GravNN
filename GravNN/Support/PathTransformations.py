import os
import sys
from pathlib import PureWindowsPath

import GravNN


def make_windows_path_posix(file):
    # If the file was saved on windows but we are running on mac, load the mac path.
    if "C:\\" in file and sys.platform.startswith("darwin"):
        old_path = PureWindowsPath(file).as_posix()
    else:
        old_path = file
    module_path = os.path.dirname(GravNN.__file__)
    file = module_path + old_path.split("/GravNN")[-1]

    return file
