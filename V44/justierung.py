# wechsle die Working Directory zum Versuchsordner, damit das Python-Script von überall ausgeführt werden kann
"""import os,pathlib
project_path = pathlib.Path(__file__).absolute().parent.parent
os.chdir(project_path)
# benutze die matplotlibrc und header-matplotlib.tex Dateien aus dem default Ordner
os.environ['MATPLOTLIBRC'] = str(project_path.parent/'default'/'python'/'matplotlibrc')
os.environ['TEXINPUTS'] =  str(project_path.parent/'default'/'python')+':'
"""
# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from uncertainties import ufloat

# Ergebnisse JSON Datei einlesen (am Anfang)
json_file_path = 'data/Ergebnisse.json'
try:
    with open(json_file_path,'r') as json_file:
        Ergebnisse = json.load(json_file)
except FileNotFoundError as err:
    Ergebnisse = {}

########################################################
## Detektor Scan Halbwertsbreite und maximale Intensität
########################################################
print('Plot: Detektor-Scan...')
angle, intensity = np.genfromtxt('data/Detektorscan1.UXD', unpack=True)

# Gaußfunktion als Ausgleichskurve
def gauss(x,a,b,sigma,mu):
    return a/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2)) + b

# mask = (angle > -0.1) & (angle != 0.3)
mask = angle == angle # dummy mask
p0 = [10**6, 0, 10**(-2), 10**(-2)]
params,pcov = curve_fit(gauss, angle[mask], intensity[mask], p0=p0)

# Ausgleichskurven Parameter abspeichern
if not 'Detektorscan' in Ergebnisse:
    Ergebnisse['Detektorscan'] = dict()
if not 'Ausgleichsrechnung' in Ergebnisse['Detektorscan']:
    Ergebnisse['Detektorscan']['Ausgleichsrechnung'] = dict()
for i,name in enumerate(['a','b','sigma[degree]','mu[degree]']):
    Ergebnisse['Detektorscan']['Ausgleichsrechnung'][name] = f'{ufloat(params[i],np.absolute(pcov[i][i])**0.5) : .2u}'

angle_linspace = np.linspace(np.min(angle),np.max(angle), 1000)
intensity_gauss = gauss(angle_linspace,*params)

# Intensitäts Maximum
I_max = np.max(intensity_gauss)
Ergebnisse['Detektorscan']['I_max_gemessen'] = f'{np.max(intensity) : .4e}'
Ergebnisse['Detektorscan']['I_max_gauss'] = f'{I_max : .4e}'

# Halbwertsbreite anhand der Ausgleichsrechnung
# Full Width Half Maximum
left_FWHM = root(lambda x: gauss(x,*params)-(I_max/2), x0=-0.01).x[0]
right_FWHM = root(lambda x: gauss(x,*params)-(I_max/2), x0=0.1).x[0]
FWHM = np.absolute(right_FWHM - left_FWHM)
Ergebnisse['Detektorscan']['Halbwertsbreite[degree]'] = f'{FWHM : .4e}'
# Wikipedia: Die Halbwertsbreite einer Normalverteilung ist das ungefähr 2,4-Fache (genau 2*sqrt(2*ln(2))) der Standardabweichung.
Ergebnisse['Detektorscan']['Halbwertsbreite_gauss[degree]'] = f'{params[2]*(2*np.sqrt(2*np.log(2))) : .4e}'

# Detektor Scan Plotten
plt.plot(angle_linspace, intensity_gauss, 'k-', label='Ausgleichskurve')
plt.plot([left_FWHM, right_FWHM], [I_max/2, I_max/2], 'b--', label='Halbwertsbreite')
plt.plot(angle, intensity, 'ro', label='Messdaten')
plt.xlabel(r'$\alpha \:/\: °$')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.legend()
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('plot_detektorscan.pdf')
plt.clf()

############
## Z-Scan 1
############
print('Plot: Z-Scan...')
z, intensity = np.genfromtxt('data/ZScan1.UXD', unpack=True)

# Strahlbreite Ablesen
i_d = [28,-12]
d0 = np.abs(z[i_d[0]]-z[i_d[1]]) # mm

if not 'Z-Scan' in Ergebnisse:
    Ergebnisse['Z-Scan'] = dict()
Ergebnisse['Z-Scan']['d_0[mm]'] = d0

# Z Scan Plotten
plt.axvline(z[i_d[0]],color='blue',linestyle='dashed',label='Strahlgrenzen')
plt.axvline(z[i_d[1]],color='blue',linestyle='dashed')
plt.plot(z, intensity, 'ro', label='Messdaten')
plt.xlabel(r'$z \:/\: mm$')
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.legend()
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('plot_zscan.pdf')
#plt.show()
plt.clf()

##################
## Rocking-Scan 1
##################
print('Plot: Rocking-Scan...')
angle, intensity = np.genfromtxt('data/RockingScan1.UXD', unpack=True)

# Geometriewinkel ablesen
i_g = [7,-6]
a_g = np.mean(np.abs(angle[i_g]))

D = 20 #mm

# Geometriewinkel brechnen aus Strahlbreite und Probenlänge
a_g_berechnet = np.rad2deg(np.arcsin(d0/D))

if not 'Rockingscan' in Ergebnisse:
    Ergebnisse['Rockingscan'] = dict()
Ergebnisse['Rockingscan']['alpha_g_l[degree]'] = angle[i_g[0]]
Ergebnisse['Rockingscan']['alpha_g_r[degree]'] = angle[i_g[1]]
Ergebnisse['Rockingscan']['alpha_g[degree]'] = a_g
Ergebnisse['Rockingscan']['alpha_g_berechnet[degree]'] = a_g_berechnet


# Rocking Scan Plotten
plt.axvline(angle[i_g[0]],color='blue',linestyle='dashed',label='Geometriewinkel')
plt.axvline(angle[i_g[1]],color='blue',linestyle='dashed')
plt.plot(angle, intensity, 'ro', label='Messdaten')
plt.xlabel(r'$\alpha \:/\: °')
plt.ylabel(r'$I \:/\:$ Hits pro Sekunde')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.legend()
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('plot_rockingscan.pdf')
#plt.show()
plt.clf()

# Ergebnisse als JSON Datei speichern (am Ende)
with open(json_file_path,'w') as json_file:
    json.dump(Ergebnisse, json_file, indent=4)