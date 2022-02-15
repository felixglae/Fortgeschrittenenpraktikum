# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
from uncertainties import ufloat
import sympy
from uncertainties import correlated_values, correlation_matrix
import scipy.integrate as int
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as sdevs
import scipy.constants as con
from scipy.constants import physical_constants as pcon

I1, B1, B2 = np.genfromtxt('data/B1.txt', unpack=True)
I2, B = np.genfromtxt('data/B2.txt', unpack=True)

def f(x, a, b):
	return a*x + b
params, cov = curve_fit(f, I2, B)
I = np.linspace(0, 17)

plt.plot(I1, B1, 'rx', label='steigend')
plt.plot(I1, B2, 'bx', label='fallend')
plt.plot(I, f(I, *params), 'k-', label='lineare Regression')
plt.xlabel(r'$I \:/\: A$')
plt.ylabel(r'$B \:/\: mT$')
plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
plt.savefig('plot.pdf')

print('Parameter: ', params, '\nFehler: ', np.sqrt(np.diag(cov)))
# Parameter:  [57.90299278 17.8245614 ]
# Fehler:  [0.39937431 3.97706759]
