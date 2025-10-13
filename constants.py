import numpy as np
mu = 2.203187099e13 #gravitational parameter
R = 2439.7e3 #radius of mercury
vec_nadir_global = np.array([0,0,1])
vec_nadir_0 = np.array([0,0,1])
h = 750e3
sun_unit_global = np.array([0,0,1])

I = np.array([
    [1032.537, 13.602, 97.717],
    [13.602, 983.014, -48.576],
    [97.717, -48.576, 400.674]
])


