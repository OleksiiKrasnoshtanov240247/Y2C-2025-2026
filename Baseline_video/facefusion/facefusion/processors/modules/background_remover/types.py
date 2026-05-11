from typing import Literal, TypedDict

from facefusion.types import Mask, VisionFrame

BackgroundRemoverInputs = TypedDict('BackgroundRemoverInputs',
{
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask
})

<<<<<<< HEAD
BackgroundRemoverModel = Literal['ben_2', 'birefnet_general', 'birefnet_portrait', 'isnet_general', 'modnet', 'ormbg', 'rmbg_1.4', 'rmbg_2.0', 'silueta', 'u2net_cloth', 'u2net_general', 'u2net_human', 'u2netp']
=======
BackgroundRemoverModel = Literal['ben_2', 'birefnet_general', 'birefnet_portrait', 'corridor_key_1024', 'corridor_key_2048', 'isnet_general', 'modnet', 'ormbg', 'rmbg_1.4', 'rmbg_2.0', 'silueta', 'u2net_cloth', 'u2net_general', 'u2net_human', 'u2netp']
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
