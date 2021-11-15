import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
import scipy.constants as scicon
import pandas as pd


def B(I,N,R):
    return mu0 * (8 / np.sqrt(125)) * I * N / R

def lin(x,m,b):
    return m * x + b

def gF(m):
    return 4 * np.pi * me / (e0 * m) *1000

def gJ(J,S,L):
    return (3.0023*J*(J+1)+1.0023*(S*(S+1)-L*(L+1)))/(2*J*(J+1))

def nucleonSpin(gJ,gF):
    return gJ / (2 * gF) - 1/2

def U(g,B):
    return g * muB * B

def U2(g,B,Mf,Ehy):
    return g**2 * muB**2 * B**2 * (1-Mf) / Ehy

def exp(x,a,b,c):
    return a * np.exp(b * x) + c

def hyp(x,a,b,c):
    return a + b / (x - c)

#Naturkonst
mu0 = scicon.mu_0
me = scicon.physical_constants['electron mass'][0]
e0 = scicon.physical_constants['elementary charge'][0]
muB = scicon.physical_constants['Bohr magneton'][0]

#Messwerte
rf_freq, IH85, IS85, IH87, IS87 = np.genfromtxt('data/RF_Frequenz.txt',unpack=True)
#rf_freq *= 1e-3
IH85 *= 3e-4
IS85 *= 1e-3
IH87 *= 3e-4
IS87 *= 1e-3

amplitude, T85, T87 = np.genfromtxt('data/Perioden.txt',unpack=True)
#T85 *= 1e-3
#T87 *= 1e-3


#Spulen
Ns = 11
Nh = 154
Nv = 20
Rs = 0.1639
Rh = 0.1579
Rv = 0.11735

Ih = 200e-3
Bv_Erde = B(Ih,Nv,Rv)
print("Erdmagnetfeld (vertikal):",Bv_Erde*1e6,"muT")

Bs85 = B(IS85,Ns,Rs)
Bh85 = B(IH85,Nh,Rh)
Bg85 = Bs85 + Bh85

Bs87 = B(IS87,Ns,Rs)
Bh87 = B(IH87,Nh,Rh)
Bg87 = Bs87 + Bh87

params85,cov85 = curve_fit(lin,rf_freq,Bg85)
errors85 = np.sqrt(np.diag(cov85))
m85 = ufloat(params85[0],errors85[0])
b85 = ufloat(params85[1],errors85[1])

params87,cov87 = curve_fit(lin,rf_freq,Bg87)
errors87 = np.sqrt(np.diag(cov87))
m87 = ufloat(params87[0],errors87[0])
b87 = ufloat(params87[1],errors87[1])

print("lin regeression Rb-87: m:",m85*1e6,"muT/kHz b:",b85*1e6)
print("lin regeression Rb-85: m:",m87*1e6,"muT/kHz b:",b87*1e6)

f_fit = np.linspace(0,1000,1000)
plt.plot(f_fit,lin(f_fit,*params85)*1e6,'r--',label="Fit Rb-87")
plt.plot(f_fit,lin(f_fit,*params87)*1e6,'g--',label="Fit Rb-85")
plt.plot(rf_freq,Bg85*1e6,'rx',label="Erster Peak")
plt.plot(rf_freq,Bg87*1e6,'gx',label="Zweiter Peak")
plt.xlabel(r"$\nu_{RF}\,/\,$kHz")
plt.ylabel(r"$B\,/\,\mu$T")
plt.grid()
plt.legend()
plt.savefig("plot1.pdf")
plt.clf()

gF85 = gF(m85)
gF87 = gF(m87)
print("gF87:",gF85)
print("gF85:",gF87)
print("Verhaeltnis gF87/gF85:",gF85/gF87)
print("Verhaeltnis gF85/gF87:",gF87/gF85)

gJ = gJ(.5,.5,0)
print("gJ:",gJ)

I85 = nucleonSpin(gJ,gF85)
I87 = nucleonSpin(gJ,gF87)

print('I87:',I85)
print('I85:',I87)

EHy85 = 2.01e-24
EHy87 = 4.53e-24
Mf = 0
print('Linear 87:',U(gF85,Bg85.max()),'J, Linear 85:',U(gF87,Bg87.max()),'J')
print('Quadratisch 87:',U2(gF85,Bg85.max(),Mf,EHy85),'J, Quadratisch 85:',U2(gF87,Bg87.max(),Mf,EHy87),'J')

df = pd.read_csv('data/Anstieg/F0015CH2.CSV')
x_anstieg = df[df.keys()[3]].to_numpy()
y_anstieg = df[df.keys()[4]].to_numpy()
y_anstieg -= max(y_anstieg)

params85, cov85 = curve_fit(exp,x_anstieg,y_anstieg)
errors85 = np.sqrt(np.diag(cov85))
a85 = ufloat(params85[0], errors85[0])
b85 = ufloat(params85[1], errors85[1])
c85 = ufloat(params85[2], errors85[2])

print("Exp Fit params:",a85,b85,c85)

x = np.linspace(min(x_anstieg),max(x_anstieg),3000)
plt.plot(x_anstieg,y_anstieg,'b--',label="Anstieg Oszilloskop")
plt.plot(x, exp(x,*params85),'r-',label="Exp. Fit")
plt.grid()
plt.legend()
plt.xlabel("Zeit in s")
plt.ylabel(r"$U - U_{Max}$")
plt.savefig('anstieg.pdf')
plt.clf()


#param_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
params85, covariance85 = curve_fit(hyp, amplitude, T85)#, p0 = (0.3, 2.7, 0.3), bounds = param_bounds)
errors85 = np.sqrt(np.diag(covariance85))
a85 = ufloat(params85[0], errors85[0])
b85 = ufloat(params85[1], errors85[1])
c85 = ufloat(params85[2], errors85[2])

params87, covariance87 = curve_fit(hyp, amplitude, T87)#, p0 = (0.15, 1.8, 0.15), bounds = param_bounds)
errors87 = np.sqrt(np.diag(covariance87))
a87 = ufloat(params87[0], errors87[0])
b87 = ufloat(params87[1], errors87[1])
c87 = ufloat(params87[2], errors87[2])

print("Perioden Fit 87:",a85,b85,c85)
print("Perioden Fit 85:",a87,b87,c87)
print('b85/b87:',b87/b85)

x = np.linspace(min(amplitude),max(amplitude),1000)
plt.plot(x,hyp(x,*params85),'b--')
plt.plot(x,hyp(x,*params87),'r--')
plt.plot(amplitude,T85,'bx',label="Rb-85")
plt.plot(amplitude,T87,'rx',label="Rb-87")
plt.grid()
plt.legend("T in ms")
plt.xlabel("U in Volt")
plt.legend()
plt.savefig("perioden.pdf")

#rf_freq, IH85, IS85, IH87, IS87
print("res_fr")
for i in range(len(rf_freq)):
    print(rf_freq[i],'&',IH85[i],'&',IS85[i],'&',IH87[i],'&',IS87[i],'\\\\')
print("Perioden")
for i in range(len(amplitude)):
    print(amplitude[i],'&',T85[i],'&',T87[i],'\\\\')