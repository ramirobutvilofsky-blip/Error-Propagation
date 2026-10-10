import numpy as np

# ------------------------------- Load data -------------------------------

# name: (values, error). Values can be a list or np.loadtxt('file.csv', delimiter=',')
VARIABLES = {
    'x1': ([0.5, 1.0, 2.0, 3.0, 4.0], 0.05),
    'x2': ([8.0, 9.0, 1.0, 7.0, 3.0], 1.0),
}

# name: (value, error)
CONSTANTS = {
    'A': (3.0, 0.1),
    'K': (0.5, 0.02),
}

function = 'A*np.exp(-K*x1)/x2'

#------------------------------------------------------------------------------
# ------------------------------- Automatization for the main function (do not edit) -------------------------------

variables_l = list(VARIABLES)
constants_l = list(CONSTANTS)

nv = len(variables_l)
nc = len(constants_l)

variables = np.array([np.asarray(vals, dtype=float) for vals, _ in VARIABLES.values()]).T
var_errors = np.array([err for _, err in VARIABLES.values()], dtype=float)

cons = [val for val, _ in CONSTANTS.values()]
cte_errors = np.array([err for _, err in CONSTANTS.values()], dtype=float)

lengths = {len(np.atleast_1d(vals)) for vals, _ in VARIABLES.values()}
if len(lengths) != 1:
    raise ValueError(f"Variables have different lengths: {lengths}")