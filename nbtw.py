import numpy as np

""" to run you need linux terminal i.e. something like
"Python 3.7.5 (default, Feb 23 2021, 13:22:40)
[GCC 8.4.0] on linux" should work with just "python nbtw_test.py".
In other case, I also attached notebook which can be run in aalto jupyter lab.
To answer the last question, there is an attached snippet named "results". """

def identity(A):
	return np.identity(A.shape[0])

def delta(A):
	d = np.sum(A, axis=1)
	delta = identity(A)
	delta[np.diag_indices_from(delta)] = d
	return delta


def recursive(A, k):
	assert(k >= 3)
	p1 = nbtw(A, 2)
	p2 = nbtw(A, 1)
	Id = identity(A) - delta(A)
	for _ in range(3, k):
		pr = np.dot(A, p1) + np.dot(Id, p2)
		p2 = p1
		p1 = pr
	return p1

def nbtw(A, k):
	if k==1:
		return A
	elif k==2:
		return np.dot(A, A) - delta(A)
	else:
		return recursive(A, k+1)

def adj_complete(n):
	return np.ones((n,n)) - np.identity(n)

def n_k(n, k):
	A = adj_complete(n)
	return nbtw(A, k)