import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

def ToBogen(phi):
    return phi*(2*np.pi/360)

def Gerade(x,a):
    return a*x

epsilon = 8.854186e-12
e = 1.602176e-19
c = 299792458
n_GaAs = 3.4

lam,undotiert_1,undotiert_2 = np.genfromtxt('data/undotiert.txt',unpack=True)
leichtdotiert_1,leichtdotiert_2 = np.genfromtxt('data/leichtdotiert.txt',unpack=True)
hochdotiert_1,hochdotiert_2 = np.genfromtxt('data/hochdotiert.txt',unpack=True)
z_bfeld,bfeld = np.genfromtxt('data/bfeld.txt',unpack=True)
z_bfeld -= z_bfeld[0]

l_undot = 5.11 #in mm
l_leichtdot = 1.36
n_leichtdot = 1.2e18
l_hochdot = 1.296
n_hochdot = 2.8e18
B = max(bfeld)*10e-3
m_e = 9.10938356e-31

theta_dif_undot = (undotiert_1-undotiert_2) / 2
theta_dif_leichtdot = (leichtdotiert_1 -leichtdotiert_2) / 2
theta_dif_hochdot = (hochdotiert_1 - hochdotiert_2) / 2

theta_norm_undot = ToBogen(theta_dif_undot)/l_undot
theta_norm_leichtdot = ToBogen(theta_dif_leichtdot)/l_leichtdot-theta_norm_undot
theta_norm_hochdot = ToBogen(theta_dif_hochdot)/l_hochdot-theta_norm_undot

theta_leicht_fit = [x for x in theta_norm_leichtdot if x > min(theta_norm_leichtdot)]
lam_fit_leicht = np.array([lam[i] for i in range(len(lam)) if i != 7])
params_hoch, cov_hoch = curve_fit(Gerade,lam**2,theta_norm_hochdot)
params_leicht, cov_leicht = curve_fit(Gerade,lam_fit_leicht**2,theta_leicht_fit)
lam_fit =np.linspace(1,7.5,200)

print('param hoch: %.7f +/- %.7f'%(params_hoch[0],cov_hoch[0]))
print('param leicht: %.7f +/- %.7f'%(params_leicht[0],cov_leicht[0]))

param_leicht = ufloat(params_leicht[0],cov_leicht[0])
param_hoch = ufloat(params_hoch[0],cov_hoch[0])

m_eff_leicht = (e**3*n_leichtdot*B / (param_leicht*10e9*8*np.pi**2*epsilon*c**3*n_GaAs))**0.5
print(m_eff_leicht)
print(m_eff_leicht/m_e)

m_eff_hoch = (e**3*n_hochdot*B / (param_hoch*10e9*8*np.pi**2*epsilon*c**3*n_GaAs))**0.5
print(m_eff_hoch)
print(m_eff_hoch/m_e)

plt.plot(z_bfeld,bfeld,'go')
plt.grid()
plt.xlabel('Z-Position in mm')
plt.ylabel('Feldstärke in mT')
plt.savefig('bfeld.png')
plt.clf()
print(max(bfeld))

plt.plot(lam**2,theta_norm_undot,'go')
plt.grid()
plt.xlabel(r'$\lambda^2 \, \mathrm{in} \, \mathrm{(\mu m)^2}$')
plt.ylabel(r'$\vartheta_\mathrm{norm} \, \mathrm{in} \, \mathrm{rad/mm}$')
plt.savefig('undotiert.png')
plt.clf()

plt.plot(lam**2,theta_norm_leichtdot,'bo',label="leicht dotiert")
plt.plot(lam**2,theta_norm_hochdot,'yo',label='hoch dotiert')
#plt.plot(lam[7]**2,theta_norm_leichtdot[7],'ko',label='vernachlässigter Datenpunkt')
plt.plot(lam_fit,Gerade(lam_fit,*params_leicht),'b--')
plt.plot(lam_fit,Gerade(lam_fit,*params_hoch),'y--')
plt.grid()
plt.legend()
plt.xlabel(r'$\lambda^2 \, \mathrm{in} \, \mathrm{(\mu m)^2}$')
plt.ylabel(r'$\mathrm{\Delta}\vartheta_\mathrm{norm} \, \mathrm{in} \, \mathrm{rad/mm}$')
plt.savefig('deltaTheta.png')
plt.clf()

print('magFeld')
for i in range(len(z_bfeld)):
    print(z_bfeld[i],'&',bfeld[i],'\\\\')
print('undotiert')
for i in range(len(lam)):
    print(lam[i],'&',undotiert_1[i],'&',undotiert_2[i],'&','%.3f'%theta_norm_undot[i],'\\\\')
print('leichtdotiert')
for i in range(len(lam)):
    print(lam[i],'&',leichtdotiert_1[i],'&',leichtdotiert_2[i],'&','%.3f'%theta_norm_leichtdot[i],'\\\\')
print('hochdotiert')
for i in range(len(lam)):
    print(lam[i],'&',hochdotiert_1[i],'&',hochdotiert_2[i],'&','%.3f'%theta_norm_hochdot[i],'\\\\')
