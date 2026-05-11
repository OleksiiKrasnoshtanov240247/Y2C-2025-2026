from typing import Optional

METADATA =\
{
	'name': 'FaceFusion',
	'description': 'Industry leading face manipulation platform',
<<<<<<< HEAD
	'version': '3.5.2',
=======
	'version': '3.6.0',
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
	'license': 'OpenRAIL-AS',
	'author': 'Henry Ruhs',
	'url': 'https://facefusion.io'
}


def get(key : str) -> Optional[str]:
	return METADATA.get(key)
