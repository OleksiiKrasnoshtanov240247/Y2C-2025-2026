<<<<<<< HEAD
from typing import Sequence


=======
import hashlib
from typing import Sequence


def sanitize_job_id(job_id : str) -> str:
	__job_id__ = job_id.replace('-', '')

	if __job_id__.isalnum():
		return job_id
	return hashlib.sha1(job_id.encode()).hexdigest()


>>>>>>> 66227f6a7a0189aec16363537239bf26c7d75a7a
def sanitize_int_range(value : int, int_range : Sequence[int]) -> int:
	if value in int_range:
		return value
	return int_range[0]
