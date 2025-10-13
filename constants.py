import numpy as np
#test



# Constants used in the simulation

mu = 2.23*10**10#gravitational parameter
R = 2439.7*10**3#radius of mercury
h = 700000 #orbit altitude

I = np.array([
    [1032.537, 13.602, 97.717],
    [13.602, 983.014, -48.576],
    [97.717, -48.576, 400.674]
])

vec_nadir_global = np.array([0,0,1])

