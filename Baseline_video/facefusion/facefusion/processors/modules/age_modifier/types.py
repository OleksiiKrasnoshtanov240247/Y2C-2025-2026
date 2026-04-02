from typing import Any, Literal, TypeAlias, TypedDict

from numpy.typing import NDArray

from facefusion.types import Mask, VisionFrame

AgeModifierInputs = TypedDict('AgeModifierInputs',
{
	'reference_vision_frame' : VisionFrame,
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

<<<<<<< HEAD
AgeModifierModel = Literal['styleganex_age']
=======
AgeModifierModel = Literal['fran', 'styleganex_age']
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

AgeModifierDirection : TypeAlias = NDArray[Any]
