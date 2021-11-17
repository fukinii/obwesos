import numpy as np
import matplotlib.pyplot as plt

from utils import solve_alpha

n = 1e20
T = 1
tolerance = 1e-9
alpha_vector = solve_alpha(n=n, temperature=T, tolerance=tolerance)

print("n = ", n)
print("T = ", T)
print("alpha0 = ", alpha_vector[0])
print("alpha1 = ", alpha_vector[1])
print("alpha2 = ", alpha_vector[2])
print("alpha3 = ", alpha_vector[3])
print("alpha = ", alpha_vector[4])

"""
n =  1e+20
T =  1
alpha0 =  0.6777274296481086
alpha1 =  0.32227216790697527
alpha2 =  4.0244491605662615e-07
alpha3 =  3.507453310115277e-17
alpha =  0.32227297263647114
"""

"""
n =  1e+20
T =  2
alpha0 =  0.09499995958446894
alpha1 =  0.8914426374151807
alpha2 =  0.013555680932821376
alpha3 =  1.7220675289254044e-06
alpha =  0.9185591653200533
"""

"""
n =  1e+20
T =  3
alpha0 =  0.020211588445244393
alpha1 =  0.6687604266471177
alpha2 =  0.30529005357305383
alpha3 =  0.005737931334584224
alpha =  1.2965543280296932
"""

"""
n =  1e+17
T =  1
alpha0 =  0.006441619799285644
alpha1 =  0.9931562598984117
alpha2 =  0.0004021202909394979
alpha3 =  1.136308355306942e-11
alpha =  0.9939605006239867
"""

"""
n =  1e+17
T =  2
alpha0 =  2.5933311697501066e-05
alpha1 =  0.11553579295694963
alpha2 =  0.8341287159513809
alpha3 =  0.0503095577799718
alpha =  1.9347218985229633
"""

"""
n =  1e+17
T =  3
alpha0 =  3.495666537204686e-08
alpha1 =  0.0005183905967263127
alpha2 =  0.10606111835551384
alpha3 =  0.8934204560910944
alpha =  2.8929019956460094
"""
