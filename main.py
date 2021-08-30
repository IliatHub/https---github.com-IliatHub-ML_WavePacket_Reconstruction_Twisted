def single_state_prop(Jmax, J0M0, I, P, taup, tdef):

    import math
    import numpy as np
    import scipy.sparse as sparse
    import scipy.linalg as linalg
    from scipy.sparse.linalg import expm_multiply
    from scipy.io import mmread, mmwrite

    def superindex(Jmax):
        """
        Generates a "superindex" matrix v. The first 
        column of v contains all the J's, while the 
        second one all the M's. The order is discussed
        in the pdf document.
        """
        v = np.zeros((1, 2))
        for m in range(-Jmax, Jmax+1):
            Js = np.arange(Jmax, abs(m)-1, -1)
            vtemp = np.repeat(m, Js.size)
            v = np.vstack((v, np.vstack((Js, vtemp)).T))
        return np.delete(v, 0, 0)

    def initialstate(J0M0, N, v):
        """
        Generates a vector representing
        the initial state defined by J0M0.
        """
        psi0 = np.zeros(N)
        index = np.where((v == J0M0).all(axis=1))[0]
        psi0[index] = 1
        return psi0

    def mats():
        """
        Load the matrix representations of the trig. functions
        """
        cos2 = mmread("Jmax10\cos2.mtx")
        sin2sin = mmread("Jmax10\sin2sin.mtx")
        sin2cos = mmread("Jmax10\sin2cos.mtx")
        cos2phi = mmread("Jmax10\cos2phi.mtx")
        cos = mmread("Jmax10\cos.mtx")
        return cos2.tocsc(), sin2sin.tocsc(), sin2cos.tocsc(), cos2phi.tocsc(), cos.tocsc()

    def intpropagation(N, v, psi0, P, taup, observable1, observable2, time):
        """
        Using matrix exponential to apply a single kick.
        Then, the wave function is propagated in time.
        """
        energy = (1/(2*I))*v[:, 0]*(v[:, 0]+1)
        psiplus = np.exp(-1j*energy*taup) * \
            expm_multiply(-1j * P*(mats()[0]-mats()[2]), psi0)
        psiplus = expm_multiply(-1j*P*(mats()[0]-mats()[1]), psiplus)
        wfafotime = np.exp(-1j*np.outer(energy, time))*psiplus[..., np.newaxis]
        return np.real(np.multiply(np.conj(wfafotime), observable1*wfafotime).sum(axis=0)), \
            np.real(np.multiply(np.conj(wfafotime),
                                observable2*wfafotime).sum(axis=0))

    # The basis size
    N = (Jmax+1)**2
    # Conversion factor between atomic time units and ps.
    psauconv = 2.418884*10 ** (-5)
    # Time grid in atomic units
    time = np.linspace(tdef[0], tdef[1], num=tdef[2])/psauconv
    v = superindex(Jmax)
    psi0 = initialstate(J0M0, N, v)
    # Define the observable.
    observable1 = mats()[0]
    observable2 = mats()[3]

    return intpropagation(N, v, psi0, P, taup, observable1, observable2, time)
