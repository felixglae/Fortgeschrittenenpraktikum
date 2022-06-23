import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import uncertainties.unumpy as unp
from uncertainties import ufloat

from uncertainties import correlated_values, correlation_matrix
import scipy.integrate as int
from uncertainties.unumpy import nominal_values as noms
from uncertainties.unumpy import std_devs as sdevs
import scipy.constants as con
from scipy.constants import physical_constants as pcon


def linear(x, m, b):
    return m * x + b

"""
t = np.array([-24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
n = np.array(
    [12, 32, 56, 71, 123, 157, 167, 180, 199, 183, 192, 196, 212, 218, 20 , 160, 162, 143, 133, 122, 119, 90, 85, 59, 46, 14, 6])

hoehe = np.mean(n[10:16])
links = n[0:9]
rechts = n[17:]
dt_rechts = t[17:]

# linearer Fit links und rechts
params_links, cov_links = curve_fit(linear, t[0:9], links)
errors_links = np.sqrt(np.diag(cov_links))
m = ufloat(params_links[0], errors_links[0])
b = ufloat(params_links[1], errors_links[1])
print("Steigung links: ", m)
print("y-Achsenabschnitt links: ", b)
print("Halbe Höhe: ", hoehe / 2)

params_rechts, cov_rechts = curve_fit(linear, dt_rechts, rechts)
errors_rechts = np.sqrt(np.diag(cov_rechts))
m = ufloat(params_rechts[0], errors_rechts[0])
b = ufloat(params_rechts[1], errors_rechts[1])
print("Steigung rechts: ", m)
print("y-Achsenabschnitt rechts: ", b)

# Berechnen des Schnittpunktes
x_links = np.linspace(-20, 0)
x_rechts = np.linspace(13, 28)

links_w = linear(x_links, *params_links)
rechts_w = linear(x_rechts, *params_rechts)

plt.figure(1)
plt.ylabel(r"$Rate \, (\mathrm{s})^{-1}$")
plt.xlabel(r"$\mathrm{d}t \, / \, \mathrm{ns}$")
plt.errorbar(t, n, yerr=np.sqrt(n), fmt='k.', label="Messwerte", capsize=2)
plt.plot(x_links, linear(x_links, *params_links), 'r',
         label="Regression links")
plt.plot(x_rechts, linear(x_rechts, *params_rechts), 'r',
         label="Regression rechts")
plt.axhline(y=hoehe, xmin=0.45, xmax=0.65, label="Plateau")
plt.axhline(y=hoehe / 2, xmin=0, xmax=1, color="green", linestyle="--",
            label="Halbwertsbreite")
plt.axvline(x=-1.75, color="green", linestyle="--")
plt.axvline(x=14.8, color="green", linestyle="--")
plt.ylim(0, 250)
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Plateau3.pdf")
plt.clf()
"""
# Kalibrierung
t2 = np.array([0.8, 1.8, 2.8, 3.8, 4.8, 5.8, 6.8, 7.8, 8.8, 9.8])
kanal = np.array([37, 81, 126, 171, 216, 261, 306, 350, 395, 440])
kanal_Fehler = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

params_kal, cov_kal = curve_fit(linear, kanal, t2)
errors_kal = np.sqrt(np.diag(cov_kal))
m1 = ufloat(params_kal[0], errors_kal[0])
b1 = ufloat(params_kal[1], errors_kal[1])
print("Steigung: ", m1)
print("y-Achsenabschnitt: ", b1)

x = np.linspace(0, 450)

plt.figure(2)
plt.xlabel(r'Kanal')
plt.ylabel(r"$T_{VZ} \, / \, \mathrm{\mu s}$")
plt.plot(x, linear(x, *params_kal), 'b', label="Regression")
plt.errorbar(kanal, t2, xerr=kanal_Fehler, fmt='rx', label="Messwerte")
# plt.xlim(0, 230)
plt.grid()
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("Kanal.pdf")
plt.clf()

# Bestimmung des Untergrunds
messdauer = 272190  # Sekunden
Startimpulse = 3256768
Startimpulse = ufloat(Startimpulse, np.sqrt(Startimpulse))
N = Startimpulse / messdauer
Ts = 10 * 10 ** (-6)  # Sekunden

Nf = Startimpulse * N * Ts * unp.exp(-N * Ts)
Nf_kanal = Nf / 460
Ts_real = (m1 * 460 + b1) * 10 ** (-6)
Nf_real = Startimpulse * N * Ts_real * unp.exp(-N * Ts_real)
Nf_kanal_real = Nf_real / 460
P= N * Ts_real * unp.exp(-N * Ts_real)
print("-------------------")
print(m1)
print(b1)
print(P)
print("Startimpulse: ", Startimpulse)
print("Ereignisrate: ", N)
print("Suchzeit: ", Ts)
print("Fehlmessungen: ", Nf)
print("Untergrundrate: ", Nf_kanal)
print("reale Suchzeit: ", Ts_real)
print("reale Fehlmessungen: ", Nf_real)
print("reale Untergrundrate: ", Nf_kanal_real)


# Einlesen der Daten für die Myonlebensdauer
Daten = np.genfromtxt('data/v01-messwerte.Spe', skip_header=7, skip_footer=1, dtype='int32')
bkg = 3.509
ch = np.linspace(1, 462, 461)

# Liste aus Daten[i] mal den channel wert
daten = []
for i in range(1, 462):
    daten.append(Daten[i])


# # plt.step(ch, Daten, where='mid', label='Messwerte')
# plt.hist(daten, bins=100, label='Messwerte')
# plt.xlabel('Kanal')
# plt.ylabel('Ereignisse')
# plt.xlim(0,431)
# plt.legend(loc='best', numpoints=1)
# plt.tight_layout()
# plt.savefig('plots/spektrum1.pdf')
# plt.clf()


def exp(t, N, tau, U):
    return N * np.exp(-t / tau) + U


print('''
    daten shape: {}
    ch shape: {}
'''.format(np.shape(daten), np.shape(ch)))
params, cov = curve_fit(exp, linear(np.floor(ch), noms(m1), noms(b1)), daten)
errors = np.sqrt(np.diag(cov))
N = ufloat(params[0], errors[0])
tau = ufloat(params[1], errors[1])
U = ufloat(params[2], errors[2])
ambda = 1 / tau
print(ambda)
print('''
	Ergebnisse des ungewichteten Fits:
	---------------------------------------------
	tau = {:.3f} micro seconds
	N = {}
	U = {}
'''.format(tau, N, U))

x = np.linspace(0, linear(509, noms(m1), noms(b1)), 1000)

plt.errorbar(linear(ch, noms(m1), noms(b1)), daten, yerr=np.sqrt(daten),
             fmt='k.',
             ms=5,
             mew=0.6,
             capsize=2,
             elinewidth=0.6,
             capthick=0.6,
             label='Messwerte')
plt.plot(x, exp(x, noms(N), noms(tau), noms(U)), label='Fit', color='r')
plt.xlabel('Zeit')
plt.ylabel('Ereignisse')
plt.grid()
plt.xlim(0, linear(469, noms(m1), noms(b1)))
plt.legend(loc='best', numpoints=1)
plt.tight_layout()
plt.savefig('spektrum1_fit.pdf')
plt.clf()

Summe = Daten.sum()
print(Summe)