<<<<<<< HEAD
from typing import List, Sequence
=======
from typing import List, Sequence, get_args
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

from facefusion.common_helper import create_float_range, create_int_range
from facefusion.processors.modules.face_enhancer.types import FaceEnhancerModel

<<<<<<< HEAD
face_enhancer_models : List[FaceEnhancerModel] = [ 'codeformer', 'gfpgan_1.2', 'gfpgan_1.3', 'gfpgan_1.4', 'gpen_bfr_256', 'gpen_bfr_512', 'gpen_bfr_1024', 'gpen_bfr_2048', 'restoreformer_plus_plus' ]
=======
face_enhancer_models : List[FaceEnhancerModel] = list(get_args(FaceEnhancerModel))
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

face_enhancer_blend_range : Sequence[int] = create_int_range(0, 100, 1)

face_enhancer_weight_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
