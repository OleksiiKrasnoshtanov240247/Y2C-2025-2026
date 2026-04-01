#!/usr/bin/env python3

import os

os.environ['OMP_NUM_THREADS'] = '1'

<<<<<<< HEAD
from facefusion import core

if __name__ == '__main__':
=======
from facefusion import conda, core

if __name__ == '__main__':
	conda.setup()
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
	core.cli()
