<<<<<<< HEAD
from facefusion.execution import create_inference_session_providers, get_available_execution_providers, has_execution_provider
=======
from facefusion.execution import create_inference_providers, get_available_execution_providers, has_execution_provider
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a


def test_has_execution_provider() -> None:
	assert has_execution_provider('cpu') is True
	assert has_execution_provider('openvino') is False


def test_get_available_execution_providers() -> None:
	assert 'cpu' in get_available_execution_providers()


<<<<<<< HEAD
def test_create_inference_session_providers() -> None:
	inference_session_providers =\
=======
def test_create_inference_providers() -> None:
	inference_providers =\
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
	[
		('CUDAExecutionProvider',
		{
			'device_id': 1,
			'cudnn_conv_algo_search': 'EXHAUSTIVE'
		}),
		'CPUExecutionProvider'
	]

<<<<<<< HEAD
	assert create_inference_session_providers(1, [ 'cpu', 'cuda' ]) == inference_session_providers
=======
	assert create_inference_providers(1, [ 'cpu', 'cuda' ]) == inference_providers
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
