from main import single_state_prop
import math
import matplotlib.pyplot as plt
import numpy as np

Jmax = 50
N = (Jmax+1)**2
I, dalpha = (280208, 35)
# Conversion factor between atomic time units and ps.
psauconv = 2.418884*10**(-5)
E0, tau, taup = (0.037, 0.02/psauconv, 0.05/psauconv)
P = 0.125*dalpha*math.sqrt(math.pi/math.log(16))*E0**2*tau
print(P)
# Define time grid in picoseconds
tdef = (0, 120, 500)  # (initial time, final time, number of time steps)
J0M0 = (0, 0)
# Time grid in picosecodns
time = np.linspace(tdef[0], tdef[1], num=tdef[2])
# Generating the two observables
signal1 = single_state_prop(Jmax, J0M0, I, P, taup, tdef)[0]
signal2 = single_state_prop(Jmax, J0M0, I, P, taup, tdef)[1]

plt.plot(time, signal1)
plt.plot(time, signal2)
plt.show()
