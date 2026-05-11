<<<<<<< HEAD
from typing import List, Sequence
=======
from typing import List, Sequence, get_args
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

from facefusion.common_helper import create_int_range
from facefusion.processors.modules.age_modifier.types import AgeModifierModel

<<<<<<< HEAD
age_modifier_models : List[AgeModifierModel] = [ 'styleganex_age' ]
=======
age_modifier_models : List[AgeModifierModel] = list(get_args(AgeModifierModel))
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

age_modifier_direction_range : Sequence[int] = create_int_range(-100, 100, 1)
