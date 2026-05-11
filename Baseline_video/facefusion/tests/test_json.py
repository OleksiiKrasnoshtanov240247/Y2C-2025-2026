<<<<<<< HEAD
=======
import os
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
import tempfile

from facefusion.json import read_json, write_json


def test_read_json() -> None:
<<<<<<< HEAD
	_, json_path = tempfile.mkstemp(suffix = '.json')
=======
	file_descriptor, json_path = tempfile.mkstemp(suffix = '.json')
	os.close(file_descriptor)
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

	assert not read_json(json_path)

	write_json(json_path, {})

	assert read_json(json_path) == {}


def test_write_json() -> None:
<<<<<<< HEAD
	_, json_path = tempfile.mkstemp(suffix = '.json')
=======
	file_descriptor, json_path = tempfile.mkstemp(suffix = '.json')
	os.close(file_descriptor)
>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a

	assert write_json(json_path, {})
