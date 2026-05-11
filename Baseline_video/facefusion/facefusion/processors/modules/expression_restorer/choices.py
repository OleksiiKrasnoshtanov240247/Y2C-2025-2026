<<<<<<< HEAD
from typing import List, Sequence
=======
from typing import List, Sequence, get_args
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

from facefusion.common_helper import create_int_range
from facefusion.processors.modules.expression_restorer.types import ExpressionRestorerArea, ExpressionRestorerModel

<<<<<<< HEAD
expression_restorer_models : List[ExpressionRestorerModel] = [ 'live_portrait' ]

expression_restorer_areas : List[ExpressionRestorerArea] = [ 'upper-face', 'lower-face' ]
=======
expression_restorer_models : List[ExpressionRestorerModel] = list(get_args(ExpressionRestorerModel))

expression_restorer_areas : List[ExpressionRestorerArea] = list(get_args(ExpressionRestorerArea))
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

expression_restorer_factor_range : Sequence[int] = create_int_range(0, 100, 1)
