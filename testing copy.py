import os
from numpy import array, matrix
from scipy.sparse import csr_matrix, lil_matrix
import scipy.sparse as sparse
from scipy.io import mmread, mmwrite

path = os.getcwd()
print(path)
B = mmread("cos2.mtx")
print(type(B))


def costheta():
    """
        Load the matrix representation
        of cos^2(theta).
        """
    cos2 = mmread("cos2.mtx")
    sin2sin = mmread("sin2sin.mtx")
    sin2cos = mmread("sin2cos.mtx")
    return cos2.tocsc(), sin2sin.tocsc(), sin2cos.tocsc()


print(costheta()[1])
