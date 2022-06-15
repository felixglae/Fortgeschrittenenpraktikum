import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from scipy.constants import R


Rp, Rz, U, I, t = np.genfromtxt('data/messung.txt',unpack=True)
"""Rp5, Rz5, U5, I5 = np.genfromtxt('data/messung2.txt',unpack=True)"""

"""t = np.ones(len(Rp2))*150"""
"""t5 = np.ones(len(Rp5))*30"""

M = 0.0635 # molare Masse
m = 0.342 # Masse

## combine 2 arrays

"""Rp = np.append(Rp2,Rp5)
Rz = np.append(Rz2,Rz5)
U = np.append(U2,U5)
I = np.append(I2,I5)
t = np.append(t2,t5)"""

#print(Rp2)
#print(Rp)

#for Rp, Rz, U, I, t in [[Rp2,Rz2,U2,I2,t2],[Rp5,Rz5,U5,I5,t5]]:

#    I = I*1e-3

## add estimated uncertainties to the data

Rp = unp.uarray(Rp, np.ones(len(Rp))*0.1)
Rz = unp.uarray(Rz, np.ones(len(Rz))*0.1)
U = unp.uarray(U, np.ones(len(U))*0.01)
I = unp.uarray(I, np.ones(len(I))*0.1)*1e-3
t = unp.uarray(t, np.ones(len(t))*5)

#print(Rp)

## calculate temperature (Kelvin) from resistance

def R_to_T(R):
    return (0.00134*R**2 + 2.296*R - 243.02) +273.15

Tp = R_to_T(Rp)
Tz = R_to_T(Rz)

#print(Tp)

## calculate differnces of temperatures

T_first = Tp[1:]
T_last = Tp[:-1]

T_delta = -(T_last - T_first)

#print(T_first)
#print(T_last)
#print(T_delta)

## delete last elements because there are just n-1 differnces

Tp = Tp[:-1]
Tz = Tz[:-1]
U = U[:-1]
I = I[:-1]
t = t[:-1]

## calculate C_p

#print(T_delta[0])
#print(U[0])
#print(I[0])
#print(t[0])

Cp = (U*I*t*M)/(T_delta*m)

#print(Cp)

## remove value because it timesteps changed from 2.5 to 5 min

"""Cp = np.delete(Cp, 23)
Tp = np.delete(Tp, 23)
Tz = np.delete(Tz, 23)
U = np.delete(U, 23)
I = np.delete(I, 23)
t = np.delete(t, 23)"""

## plot C_p

plt.errorbar(unp.nominal_values(Tp),unp.nominal_values(Cp), yerr=unp.std_devs(Cp), fmt='k.', capsize=3, label=r'$C_p$')
plt.plot([105,310], [3*R , 3*R], "r-", label=r'$3 R$' )
plt.xlim(110,299)
plt.xlabel(r'$T$ / K')
plt.ylabel(r'$C_p$ / $J{mol}^{-1}K^{-1}$')
plt.legend(loc='best')
plt.savefig('plot1.pdf')
plt.clf()

## extrapolate alpha

T_alpha = np.linspace(70, 300, 24)
alpha = np.array([
                7,8.5,9.75,10.7,11.5,12.1,12.65,13.15,
                13.6,13.9,14.25,14.5,14.75,14.95,15.2,15.4,
                15.6,15.75,15.9,16.1,16.25,16.35,16.5,16.65])*1e-6

#print(len(Tp))
#print(T_alpha)
#print(unp.nominal_values(Tp))
#print(alpha)

def lin(x,a,b):
    return a*x+b

var1, cov1 = curve_fit(lin, T_alpha[0:7], alpha[0:7])
errs1 = np.sqrt(np.diag(cov1))

plt.plot(T_alpha[0:7], alpha[0:7]*1e6, "b.")
plt.plot(T_alpha[0:7], lin(T_alpha[0:7], *var1)*1e6, "b-")
plt.plot(unp.nominal_values(Tp[0:12]), lin(unp.nominal_values(Tp[0:12]), *var1)*1e6, "k.")

var2, cov2 = curve_fit(lin, T_alpha[7:], alpha[7:])
errs2 = np.sqrt(np.diag(cov2))

plt.plot(T_alpha[7:], alpha[7:]*1e6, "r.")
plt.plot(T_alpha[7:], lin(T_alpha[7:], *var2)*1e6, "r-")
plt.plot(unp.nominal_values(Tp[12:]), lin(unp.nominal_values(Tp[12:]), *var2)*1e6, "k.")

plt.savefig('plot2.pdf')
plt.clf()

## calculate C_v

Cv = Cp[0:18] - 9*lin(Tp[0:18], *var1)**2*137.8*1e9*7.09*1e-6*Tp[0:18]
"""Cv2 = Cp[12:] - 9*lin(Tp[12:], *var2)**2*137.8*1e9*7.09*1e-6*Tp[12:]

Cv = np.append(Cv1,Cv2)"""

#print(Cv)

## plot C_v

plt.errorbar(unp.nominal_values(Tp),unp.nominal_values(Cv), yerr=unp.std_devs(Cv), fmt='k.', capsize=3, label=r'$C_v$')
#plt.plot(unp.nominal_values(Tp),unp.nominal_values(Cp), 'b.', label=r'$C_p$')
plt.plot([105,310], [3*R , 3*R], "r-", label=r'$3 R$' )
plt.xlim(110,299)
plt.xlabel(r'$T$ / K')
plt.ylabel(r'$C_v$ / $J{mol}^{-1}K^{-1}$')
plt.legend(loc='best')
plt.savefig('plot3.pdf')
plt.clf()

## Tabellen

count = 0
while(count < len(Tp)):
    print(T_delta[count], "&", U[count], "&", I[count], "&", t[count], "&", Cp[count], "\\\\")
    count += 1

print()

count = 0
while(count < len(Tp[0:12])):
    print(Tp[count], "&", Cp[count], "&", lin(Tp[count], *var1), "&", Cv[count], "\\\\")
    count += 1

while(count < len(Tp)):
    print(Tp[count], "&", Cp[count], "&", lin(Tp[count], *var1), "&", Cv[count], "\\\\")
    count += 1

print()

quo = np.array([2.6, 2.6, 1.9, 1.2, 1.6, 1.7, 0.9, 1.7, 0.9, 1.5, 0.7, 1.6, 0.9, 0.9, 0.9, 0.9, 0.9, 1.8, 1.3, 1.8, 1.8, 1.8])
deby_temp = []
count = 0
while(count < len(Tp[0:6])):
    deby_temp = np.append(deby_temp, quo[count]*Tp[count])
    print(Cv[count], "&",quo[count], "&", Tp[count], "&", quo[count]*Tp[count], "\\\\")
    count += 1

print(len(Cp))
print(np.mean(deby_temp))
print(np.mean(Cp))
###
