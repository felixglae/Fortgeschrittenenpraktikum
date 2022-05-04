"""# wechsle die Working Directory zum Versuchsordner, damit das Python-Script von überall ausgeführt werden kann
import os,pathlib
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
import matplotlib as mpl
from scipy.optimize import curve_fit, root
from scipy.signal import find_peaks
from scipy.stats import sem
from uncertainties import ufloat

##################
## Daten Einlesen
##################
# Ergebnisse JSON Datei einlesen (am Anfang)
json_file_path = 'data/Ergebnisse.json'
try:
    with open(json_file_path, 'r') as json_file:
        Ergebnisse = json.load(json_file)
except FileNotFoundError as err:
    Ergebnisse = {}

if not 'Messung' in Ergebnisse:
    Ergebnisse['Messung'] = dict()

# reflectivity scan:
a_refl, I_refl = np.genfromtxt('data/omega2thetascan.UXD', unpack=True)
# diffuse scan
a_diff, I_diff = np.genfromtxt('data/diffusscan.UXD', unpack=True)


# angles are equal for a_refl and a_diff
a = a_refl

# cut off end data points
a_min = 0
a_max = 1.7
mask = (a >= a_min) & (a < a_max)
a = a[mask]
I_refl = I_refl[mask]
I_diff = I_diff[mask]

#################
## Berechnungen
#################
# Eingehende Intensität als das Maximum vom Detektorscan
# aber mit 5 multipliziert weil nun statt 1s 5s pro Winkel gemessen wurden


I_0 = float(Ergebnisse['Detektorscan']['I_max_gauss']) * 5      #5s per angle

# reflectivity: R=I_r/I_0
R_refl = I_refl / I_0
R_diff = I_diff / I_0

# subtract diffuse scan
R = R_refl - R_diff

# geometry angle
a_g = float(Ergebnisse['Rockingscan']['alpha_g[degree]'])

# beam width
d_0 = float(Ergebnisse['Z-Scan']['d_0[mm]'])
D = 20  # mm

# geometry factor
G = np.ones_like(R)
G[a < a_g] = D * np.sin(np.deg2rad(a[a < a_g])) / d_0

# adjust R with G
R_G = R / G

# ideal Fresnel reflectivity
a_c_Si = 0.223
R_ideal = (a_c_Si / (2 * a)) ** 4

# find peaks
# curve fit for find_peaks
peaks_mask = (a >= 0.3) & (a <= 1.2)


def f(x, b, c):
    return b * x + c


params, pcov = curve_fit(f, a[peaks_mask], np.log(R_G[peaks_mask]))
R_fit = np.exp(f(a[peaks_mask], *params))

# Minima der Kissig-Oszillation finden
i_peaks, peak_props = find_peaks(-(R_G[peaks_mask] - R_fit), distance=7)
i_peaks += np.where(peaks_mask)[0][0]

# Schichtdicke bestimmen
lambda_ = 1.54 * 10 ** (-10)  # m

delta_a = np.diff(np.deg2rad(a[i_peaks]))
delta_a_mean = ufloat(np.mean(delta_a), sem(delta_a))

d = lambda_ / (2 * delta_a_mean)

Ergebnisse['Messung']['delta_a_mean[degree]'] = f'{delta_a_mean:.2u}'
Ergebnisse['Messung']['d[m]'] = f'{d:.2u}'

# Parratt algorithm

# save R_G und a_i for interactive plot
np.savetxt('R_G.csv', list(zip(a, R_G)), header='a_i,R_G', fmt='%.4f,%.10e')

# a_i angle of incidence
# n refractive index
# n1 air, n2 nano-film, n3 substrat
# sigma roughness
# sigma1 nano-film, sigma2 substrat
# z1=0, z2 layer thickness
# k=2pi/lambda value of wave vector
# constants:
n1 = 1.
z1 = 0.
k = 2 * np.pi / lambda_

# parameters to adjust R_parr to RG
delta2 = 0.55 * 10 ** (-6)
delta3 = 6.6 * 10 ** (-6)
sigma1 = 7.5 * 10 ** (-10)  # m
sigma2 = 6.3 * 10 ** (-10)  # m
z2 = 8.6 * 10 ** (-8)  # m


def parrat_rau(a_i, delta2, delta3, sigma1, sigma2, z2):
    n2 = 1. - delta2 + 0.49 * 10 ** (-8) * 1.j
    n3 = 1. - delta3 + 1.72 * 10 ** (-7) * 1.j

    a_i = np.deg2rad(a_i)

    kz1 = k * np.emath.sqrt(n1 ** 2 - np.cos(a_i) ** 2)
    kz2 = k * np.emath.sqrt(n2 ** 2 - np.cos(a_i) ** 2)
    kz3 = k * np.emath.sqrt(n3 ** 2 - np.cos(a_i) ** 2)

    r12 = (kz1 - kz2) / (kz1 + kz2) * np.exp(-2 * kz1 * kz2 * sigma1 ** 2)
    r23 = (kz2 - kz3) / (kz2 + kz3) * np.exp(-2 * kz2 * kz3 * sigma2 ** 2)

    x2 = np.exp(-2.j * kz2 * z2) * r23
    x1 = (r12 + x2) / (1 + r12 * x2)
    R_parr = np.abs(x1) ** 2

    return R_parr


params = [delta2, delta3, sigma1, sigma2, z2]

R_parr = parrat_rau(a, *params)

# critical angle
a_c2 = np.rad2deg(np.sqrt(2 * delta2))
a_c3 = np.rad2deg(np.sqrt(2 * delta3))

Ergebnisse['Messung']['a_c2[degree]'] = a_c2
Ergebnisse['Messung']['a_c3[degree]'] = a_c3

R_ideal[a <= a_c2] = np.nan
R_parr[a <= a_c2] = np.nan

############
## Plotten
############
# Reflektivitäts Scan Plotten
print('Plot: Mess-Scan...')
mpl.rcParams['lines.linewidth'] = 0.9
mpl.rcParams['axes.grid.which'] = 'major'
plt.axvline(a_c2, linewidth=0.6, linestyle='dashed', color='blue', label=r'$\alpha_\mathrm{c,PS},\alpha_\mathrm{c,Si}$')
plt.axvline(a_c3, linewidth=0.6, linestyle='dashed', color='blue')
plt.plot(a, R_refl, '-', color='black', label='Reflektivitätsscan')
plt.plot(a, R_diff, '-', label='Diffuser Scan')
plt.plot(a, R, '-', label='Reflektivitätsscan - Diffuser Scan')
# plt.plot(a[peaks_mask],R_fit, '--', label='Peaks Curve Fit')
plt.xlabel(r'$\alpha_i \:/\:°$')
plt.ylabel(r'$R$')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 8})
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('plot_messung.pdf')
plt.show()
plt.clf()

mpl.rcParams['lines.linewidth'] = 0.9
mpl.rcParams['axes.grid.which'] = 'major'
plt.plot(a, R_ideal, '-', color='pink', label='Fresnelreflektivität von Si')
plt.plot(a, R_parr, '-', label='Theoriekurve')
plt.plot(a, R_G, '-', label=r'(Reflektivitätsscan - Diffuser Scan)$ \, / \, G $')
plt.plot(a[i_peaks], R_G[i_peaks], 'kx', label='Oszillationsminima', alpha=0.8)
# plt.plot(a[peaks_mask],R_fit, '--', label='Peaks Curve Fit')
plt.xlabel(r'$\alpha_i \:/\:°$')
plt.ylabel(r'$R$')
plt.yscale('log')
plt.legend(loc='upper right', prop={'size': 8})
plt.tight_layout(pad=0.15, h_pad=1.08, w_pad=1.08)
plt.savefig('plot_messung2.pdf')
plt.show()
plt.clf()
##########
# Ergebnisse als JSON Datei speichern (am Ende)
with open(json_file_path, 'w') as json_file:
    json.dump(Ergebnisse, json_file, indent=4)