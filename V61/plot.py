import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import pandas as pd


### Stabilitätsbed.
def stab(L, r1, r2):
    return (1-L/r1)*(1-L/r2)

def stab_inf(L, r1):
    return 1-L/r1

L = np.linspace(-0.5,3.0,100)

r1, r2 = 1.4, 1.4
plt.plot(L, stab(L, r1, r2), label=r"Anordnung 1 ($r_1 = 1,4$ $r_2 = 1,4$)")
r1, r2 = 1, 1.4
plt.plot(L, stab(L, r1, r2), label=r"Anordnung 2 ($r_1 = 1$ $r_2 = 1,4$)")
r1  = 1.4
plt.plot(L, stab_inf(L, r1), label=r"Anordnung 3 ($r_1 = 1,4$ $r_2 = \infty$)")


plt.grid()
plt.legend()
plt.xlabel(r"$L/m$")
plt.ylabel(r"$g_1 g_2$")
plt.xlim(-0.5, 3.0)
plt.ylim(-1.5, 2.5)
plt.savefig('plot1.pdf')
plt.clf()


### TEM

x00, I00 = np.genfromtxt('data/tem00.txt',unpack=True)
x01, I01 = np.genfromtxt('data/tem01.txt',unpack=True)
x02, I02 = np.genfromtxt('data/tem02.txt',unpack=True)

def tem00(x, a, b, c):
    return a*np.exp(-2*(x-b)**2/c**2)

def tem01(x, a, b, c):
    return a*4*(x-b)**2/c**2*np.exp(-2*(x-b)**2/c**2)

def tem02(x, a, b, c):
    return a*(8*(x-b)**2/c**2-2)**2*np.exp(-2*(x-b)**2/c**2)

var00, cov00 = curve_fit(tem00, x00, I00)
errs00 = np.sqrt(np.diag(cov00))

var01, cov01 = curve_fit(tem01, x01, I01, [0.3, 2.2, 1.5])
errs01 = np.sqrt(np.diag(cov01))

var02, cov02 = curve_fit(tem02, x02, I02, [0.5, 4, 7])
errs02 = np.sqrt(np.diag(cov02))

print("TEM")

print(var00[0], "\\pm", errs00[0])
print(var00[1], "\\pm", errs00[1])
print(var00[2], "\\pm", errs00[2])

print()

print(var01[0], "\\pm", errs01[0])
print(var01[1], "\\pm", errs01[1])
print(var01[2], "\\pm", errs01[2])

print()

print(var02[0], "\\pm", errs02[0])
print(var02[1], "\\pm", errs02[1])
print(var02[2], "\\pm", errs02[2])

print()

xplot00 = np.linspace(-7,13.0,100)
xplot01 = np.linspace(-11,15,100)
xplot02 = np.linspace(-11,15.0,100)

plt.plot(x00, I00, "k.", label="Messdaten")
plt.plot(xplot00, tem00(xplot00, *var00), label="Fit")
plt.grid()
plt.legend()
plt.xlabel(r"x-Richtung /mm")
plt.ylabel(r"$\mathrm{I}\,/\,\mu$A")
plt.xlim(-7, 13)
plt.savefig('plot2.pdf')
plt.clf()

plt.plot(x01, I01, "k.", label="Messdaten")
plt.plot(xplot01, tem01(xplot01, *var01), label="Fit")
plt.grid()
plt.legend()
plt.xlabel(r"x-Richtung /mm")
plt.ylabel(r"$\mathrm{I}\,/\,\mu$A")
plt.xlim(-11, 15)
plt.savefig('plot3.pdf')
plt.clf()

plt.plot(x02, I02, "k.", label="Messdaten")
plt.plot(xplot02, tem02(xplot02, *var02), label="Fit")
plt.grid()
plt.legend()
plt.xlabel(r"x-Richtung /mm")
plt.ylabel(r"$\mathrm{I}\,/\,\mu$A")
plt.xlim(-11, 15)
plt.savefig('plot4.pdf')
plt.clf()

### Polarisation

w, I = np.genfromtxt('data/pol.txt',unpack=True)

w = w/360*2*np.pi

def cosi(x, a, b):
    return a*np.cos(x+b)**2

var_p, cov_p = curve_fit(cosi, w, I)
errs_p = np.sqrt(np.diag(cov_p))

print("Polarisation")
print(var_p[0], "\\pm", errs_p[0])
print(var_p[1], "\\pm", errs_p[1])
print()

wplot = np.linspace(-0.5,3.5,100)

plt.plot(w, I, "k.", label="Messdaten")
plt.plot(wplot, cosi(wplot, *var_p), label="Fit")
plt.grid()
plt.legend()
plt.xlabel(r"Polarisationswinkel/rad")
plt.ylabel(r"$\mathrm{I}\,/\,\mu$A")
plt.xlim(-0.5,3.5)
plt.savefig('plot5.pdf')
plt.clf()

### Multimoden

Lmulti, p1, p2, p3, p4, p5, p6 = np.genfromtxt('data/multimoden.txt',unpack=True)

p_average = []

for a,f in zip(p1, p6):
    #print(f, "-", a)
    p_average.append((f-a)/5)

p_average = np.array(p_average)

print("Multimoden")
print(p_average)

p_average = scicon.c/(p_average*1e6) # umrechnen in wellenlänge

Lmulti *= 1e-2

def lin(x, a):
    return a*x

var_m, cov_m = curve_fit(lin, Lmulti, p_average)
errs_m = np.sqrt(np.diag(cov_m))

print()
print(var_m[0], "\\pm", errs_m[0])
print()

Lmultiplot = np.linspace(0.8,1.4,100)

plt.plot(Lmulti, p_average, "k.", label="Messdaten")
plt.plot(Lmultiplot, lin(Lmultiplot, *var_m), label="Fit")
plt.grid()
plt.legend()
plt.xlabel(r"Resonatorlänge $L$/m")
plt.ylabel(r"$\Delta \lambda$/m")
plt.xlim(0.8,1.4)
plt.savefig('plot6.pdf')
plt.clf()

### Wellenlänge
wel = np.array([627.53, 633.19, 625.26, 639.17])

print("wellenlänge")
print(wel.mean())
print(wel.std())
