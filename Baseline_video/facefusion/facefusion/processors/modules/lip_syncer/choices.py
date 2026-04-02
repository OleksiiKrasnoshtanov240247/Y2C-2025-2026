<<<<<<< HEAD
from typing import List, Sequence
=======
from typing import List, Sequence, get_args
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

from facefusion.common_helper import create_float_range
from facefusion.processors.modules.lip_syncer.types import LipSyncerModel

<<<<<<< HEAD
lip_syncer_models : List[LipSyncerModel] = [ 'edtalk_256', 'wav2lip_96', 'wav2lip_gan_96' ]
=======
lip_syncer_models : List[LipSyncerModel] = list(get_args(LipSyncerModel))
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

lip_syncer_weight_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
