import io
import os
import pathlib
import jal

# Filepaths
PACKAGE_ROOT = pathlib.Path(jal.__file__).resolve().parent.parent.parent
DATA_DIR = PACKAGE_ROOT /'data'
SRC_DIR = PACKAGE_ROOT /'src'/'jal'

bucket = 'incluia-jalisco'
