"""
A simple setup script to create an executable using PyQt5. This also
demonstrates the method for creating a Windows executable that does not have
an associated console.

test_pyqt5.py is a very simple type of PyQt5 application

Run the build process by running the command 'python setup.py build'

If everything works well you should find a subdirectory in the build
subdirectory that contains the files needed to run the application
"""

from __future__ import annotations

import sys

from cx_Freeze import Executable, setup

try:
    from cx_Freeze.hooks import get_qt_plugins_paths
except ImportError:
    get_qt_plugins_paths = None


#include_files = ['../res/icons/icon.ico', '../Object_Detection_Help_Guide.pdf', '../res/pages/ObjDet.ui', '../res/icons/tensorflow_icon.ico',
#                 '../res/pages/help_viewer.ui','../res/pages/eval_dialog.ui', '../res/pages/createproj.ui','C:/Users/andrew/Documents/Programming/tensorflow/', 
#                 'C:/Users/andrew/Documents/Ext/Gui2/protoc/', "../res/pages/install.ui","C:/Users/andrew/Documents/Ext/Gui2/nvvm/",
#                  '../res/pages/About.ui' ]

include_files = ['../res']
if get_qt_plugins_paths:
    # Inclusion of extra plugins (since cx_Freeze 6.8b2)
    # cx_Freeze imports automatically the following plugins depending of the
    # use of some modules:
    # imageformats, platforms, platformthemes, styles - QtGui
    # mediaservice - QtMultimedia
    # printsupport - QtPrintSupport
    for plugin_name in (
        # "accessible",
        # "iconengines",
        # "platforminputcontexts",
        # "xcbglintegrations",
        # "egldeviceintegrations",
        "wayland-decoration-client",
        "wayland-graphics-integration-client",
        # "wayland-graphics-integration-server",
        "wayland-shell-integration",
    ):
        include_files += get_qt_plugins_paths("PyQt5", plugin_name)

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None
packages = ["PyQt6", "os", "sys", 'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
            'PyQt6.QtGui', 'PyQt6.QtCore', 'logging', 'cv2', 'scipy',
            'numpy', 'matplotlib', 'PyQt6.QtWidgets']
build_exe_options = {
    # exclude packages that are not really needed
    "include_files": include_files,
    "zip_include_packages": ["PyQt6"],
    "includes": ["PyQt6", "os", "sys", 'PyQt6.QtMultimedia', 'PyQt6.QtMultimediaWidgets',
            'PyQt6.QtGui', 'PyQt6.QtCore', 'logging', 'cv2', 'scipy',
            'numpy', 'matplotlib', 'PyQt6.QtWidgets'],
    "packages": packages,
    "include_msvcr": True,
}



bdist_mac_options = {
    "bundle_name": "Test",
}

bdist_dmg_options = {
    "volume_label": "TEST",
}

bdist_msi_options = {
    "add_to_path" : True

}

executables = [Executable("../src/MainGUI.py", base=base)]


setup(
    name="XRAY",
    version="0.1",
    description="X-Ray Optics",
    options={
        "build_exe": build_exe_options,
        "bdist_mac": bdist_mac_options,
        "bdist_dmg": bdist_dmg_options,
        "bdist_msi": bdist_msi_options,
    },
    

    executables=executables,
    
)