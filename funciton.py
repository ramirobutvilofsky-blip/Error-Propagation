import numpy as np
import pandas as pd
import copy
np.set_printoptions(precision=15, floatmode='fixed')

from variables import nv, variables, variables_l, var_errors,cons, constants_l, cte_errors, nc, function

# ----------------------------------------------------------------------------------------------

for n,v in zip(constants_l,cons):

    globals()[n] = v

f = lambda *args: eval(function, globals(), {**locals(), **dict(zip(variables_l, args))})

# ----------------------------------------------------------------------------------------------------------
# ------------------------------------------ DERIVATIVE ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------

def derivative(funct,v,c):

    Q = v.copy()

    derivatives_v = np.zeros([v.shape[1],v.shape[0]])

    for i in range(v.shape[1]):

        vd_plus = np.zeros([v.shape[0],v.shape[1]])
        vd_minus = np.zeros([v.shape[0],v.shape[1]])
        lims_v = []

        for j in range(v.shape[0]):

            e = np.finfo(float).eps
            h = e**(1/3) * max(abs(v[j,i]) , 1.0)
            lims_v.append(h)

        for l in range(v.shape[0]):

            Q[l,i] = v[l,i] + lims_v[l]
            vd_plus[l] = Q[l]
            
            Q[l,i] = v[l,i] - lims_v[l]
            vd_minus[l] = Q[l]

        for t in range(v.shape[0]):

            vdp = vd_plus[t].tolist()
            vdm = vd_minus[t].tolist()

            d = (funct(*vdp) - funct(*vdm))/(2*lims_v[t])
            derivatives_v[i,t] = d

        Q = v.copy()
    
    #------------------------

    derivatives_c = []
    lims_c = []

    v = v.T

    for i in range(nc):

        variables_l.append(c[i])

    cd_plus = variables_l.copy()
    cd_minus = variables_l.copy()

    for i in range(v.shape[0]):

        cd_minus[i] = v[i].copy()
        cd_plus[i] = v[i].copy()

    for i in range(v.shape[0] , v.shape[0]+len(c)):

        cd_minus[i] = eval(cd_minus[i])
        cd_plus[i] = eval(cd_plus[i])

    for i in range(v.shape[0], v.shape[0]+len(c)):
        
        if i == v.shape[0]:
                
            e = np.finfo(float).eps
            h = e**(1/3) * max(abs(cd_minus[i]) , 1.0)

            cd_minus[i] -= h
            cd_plus[i] += h
            
            lims_c.append(h)

            dc = (funct(*cd_plus) - funct(*cd_minus))/(2*h)
            derivatives_c.append(dc)

        else:

            cd_minus[i-1] = eval(c[(i-1)-nv])
            cd_plus[i-1] = eval(c[(i-1)-nv])
            
            e = np.finfo(float).eps
            h = e**(1/3) * max(abs(cd_minus[i]) , 1.0)

            cd_minus[i] -= h
            cd_plus[i] += h
            lims_c.append(h)

            dc = (funct(*cd_plus) - funct(*cd_minus))/(2*h)

            derivatives_c.append(dc)

    n = np.array(derivatives_v)
    m = np.array(derivatives_c)
    
    return n, m

#-------------------------------------------------------------------------------------------------------
# ------------------------------------------ ERROR RESULT  ----------------------------------------------------
    
pv = derivative(f,variables,constants_l)

deriv_v = np.array(pv[0])

deriv_c = pv[1]


s = np.zeros([variables.shape[1],variables.shape[0]])
r = 0

for i in range(nv):

    for j in range(deriv_v.shape[1]):

        s[i,j] = (deriv_v[i,j]*var_errors[i])**2

S = (s.T).sum(axis=1)

for p in range(nc):

    r += (deriv_c[p]*cte_errors[p])**2

l = S+r

error = np.sqrt(l)

print(error)

# ------------------------------------------ TO CSV ----------------------------------------------------

df = pd.Series(error)

df.to_csv('errores.csv')
