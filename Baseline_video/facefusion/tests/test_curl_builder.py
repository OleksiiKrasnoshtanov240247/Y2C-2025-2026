from shutil import which

from facefusion import metadata
from facefusion.curl_builder import chain, ping, run, set_timeout


def test_run() -> None:
	user_agent = metadata.get('name') + '/' + metadata.get('version')

<<<<<<< HEAD
	assert run([]) == [ which('curl'), '--user-agent', user_agent, '--insecure', '--location', '--silent' ]
=======
	assert run([]) == [ which('curl'), '--user-agent', user_agent, '--location', '--silent' ]
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a


def test_chain() -> None:
	assert chain(
		ping(metadata.get('url')),
		set_timeout(5)
	) == [ '-I', metadata.get('url'), '--connect-timeout', '5' ]
