import os
import datetime
from argparse import ArgumentParser


class Configuration:

    # experiment version
    MAJOR_VERSION = '0'
    MINOR_VERSION = '1'
    VERSION = MAJOR_VERSION + '.' + MINOR_VERSION

    # mastering randomness
    SEED = 1

    # where extracted data live
    datafolder = os.path.join('generated', 'tmp')

    # where generated files, e.g. .mmap, slurm*, *.sav files are stored
    experimentsfolder = os.path.join('generated', VERSION)
