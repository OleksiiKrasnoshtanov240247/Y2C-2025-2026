<<<<<<< HEAD
from typing import List

from facefusion.processors.modules.face_debugger.types import FaceDebuggerItem

face_debugger_items : List[FaceDebuggerItem] = [ 'bounding-box', 'face-landmark-5', 'face-landmark-5/68', 'face-landmark-68', 'face-landmark-68/5', 'face-mask' ]
=======
from typing import List, get_args

from facefusion.processors.modules.face_debugger.types import FaceDebuggerItem

face_debugger_items : List[FaceDebuggerItem] = list(get_args(FaceDebuggerItem))
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
