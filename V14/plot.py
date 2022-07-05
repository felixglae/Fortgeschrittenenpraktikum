# coding=utf-8
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(precision=4)

# Dateneinlesen
counts = np.genfromtxt('data/Alu_leer/I0.Spe', skip_header=12,
                       skip_footer=15)
counts_0 = np.genfromtxt('data/leer.Spe', skip_header=12,
                         skip_footer=17)
I_l, c_l, t_l = np.genfromtxt('data/Alu_l.txt', unpack=True)
I_2, c_2, t_2 = np.genfromtxt('data/W2.txt', unpack=True)
I_3, c_3, t_3 = np.genfromtxt('data/W3.txt', unpack=True)
I_5, c_5, t_5 = np.genfromtxt('data/W5.txt', unpack=True)

leer = counts
error_leer = np.sqrt(counts)
null = counts_0
error_null = np.sqrt(counts_0)
# df = pd.DataFrame(n)

# Geometriematrix
b = np.sqrt(2)

# A wird ja mit den Absorptionskoeffizienten multipliziert, um dann die Raten
# auszugeben. Demnach entsprechen die Komponenten der einzelnen Zeilen
# den Strecken durch die Unter-Würfel, die bei der jeweiligen Projektion
# durchstrahlt werden. Daher lassen sich die Zeilen bei uniformer Verteilung
# auch aufsummieren (wie bei Würfel 2 und 3).
A = np.matrix([[1, 1, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 1],
               [1, 0, 0, 1, 0, 0, 1, 0, 0],
               [0, 1, 0, 0, 1, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 1, 0, 0, 1],
               [b, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, b, 0, b, 0, 0, 0, 0, 0],
               [0, 0, b, 0, b, 0, b, 0, 0],
               [0, 0, 0, 0, 0, b, 0, b, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, b],
               [0, 0, 0, 0, 0, 0, b, 0, 0],
               [0, 0, 0, b, 0, 0, 0, b, 0],
               [b, 0, 0, 0, b, 0, 0, 0, b],
               [0, b, 0, 0, 0, b, 0, 0, 0],
               [0, 0, b, 0, 0, 0, 0, 0, 0]])

# Geometriematrix für Würfel 2 (vollständig aus Aluminium)
A_2 = np.matrix([[3],
                 [3 * b],
                 [2 * b],
                 [b]])
# Geometriematrix für Würfel 3 (vollständig aus Blei)
A_3 = np.matrix([[3],
                 [3 * b],
                 [2 * b],
                 [b]])
A_4 = np.matrix([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 1, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0, 1, 0, 0, 1],
                 [0, b, 0, b, 0, 0, 0, 0, 0],
                 [0, 0, b, 0, b, 0, b, 0, 0],
                 [0, 0, 0, 0, 0, b, 0, b, 0],
                 [0, b, 0, 0, 0, b, 0, 0, 0],
                 [b, 0, 0, 0, b, 0, 0, 0, b],
                 [0, 0, 0, b, 0, 0, 0, b, 0]])


def kleinsteQuadrate(y, W, A):
    temp = np.dot(np.linalg.inv(np.dot(A.T, np.dot(W, A))), A.T)
    a = np.dot(temp, np.dot(W, y))
    a_err = np.linalg.inv(np.dot(A.T, np.dot(W, A)))
    return a, np.sqrt(np.diag(a_err))


# Alu_leer
x = np.linspace(0, 511 * 662 / 63, 511)
plt.bar(x, leer, yerr=error_leer)
plt.xlim(0, 250 * 662 / 176)
plt.title('Messung des leeren Wuerfels')
plt.xlabel('Energie / keV')
plt.ylabel('Ereignisse')
plt.savefig('Alu_leer.pdf')
plt.close()

print('Spektrum der Alu_leer geplottet...')

# Leer
x = np.linspace(0, 511 * 662 / 178, 511)
plt.bar(x, null, yerr=error_null)
plt.xlim(0, 250 * 662 / 178)
plt.title('Messung ohne Wuerfel')
plt.xlabel('Energie / keV')
plt.ylabel('Ereignisse')
plt.savefig('leer.pdf')
plt.close()

print('(Zusätzlich: Spektrum der Leer geplottet...)')

c_no = 49361
t_no = 300  # s
rate_no = c_no / t_no

print('Messung ohne Wuerfel mit Rate: {:2f} +- {:2f} counts/s'
      .format(rate_no, np.sqrt(c_no) / t_no))

# I_0s für die Projektionen 2
rate_l = c_l / t_l
err_rate_l = np.sqrt(c_l) / t_l

# Würfel 2 (2, 9, 10, 11)
rate_2 = c_2 / t_2
err_rate_2 = np.sqrt(c_2) / t_2

# Würfel 3 (2, 9, 10, 11)
rate_3 = c_3 / t_3
err_rate_3 = np.sqrt(c_3) / t_3

# Würfel 5 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
rate_5 = c_5 / t_5
err_rate_5 = np.sqrt(c_5) / t_5

# Mitteln für leeren Würfel
I_leer = np.array(np.zeros(len(rate_5)))
err_I_leer = np.array(np.zeros(len(rate_5)))

for i in range(len(rate_5)):
    if (i == 9) or (i == 14):
        I_leer[i] = rate_l[1]
        err_I_leer[i] = err_rate_l[1]
    if i < 7:
        I_leer[i] = rate_l[0]
        err_I_leer[i] = err_rate_l[0]
    if (i == 13) or (i == 8) or (i == 15) or (i == 10):
        I_leer[i] = rate_l[2]
        err_I_leer[i] = err_rate_l[2]
    if (i == 7) or (i == 11) or (i == 12) or (i == 16):
        I_leer[i] = rate_l[3]
        err_I_leer[i] = err_rate_l[3]

# Umrechnen in ln(I_0/N_j)
# Proj_1 steht für die Projektionen, die bei Würfel 2 und 3 verwendet wurden
I_Proj_2 = np.array(rate_l)
err_I_Proj_2 = np.array(err_rate_l)

I_Proj_3 = np.array([rate_l[0], rate_l[1], rate_l[2], rate_l[3]])
err_I_Proj_3 = np.array([err_rate_l[0], err_rate_l[1], err_rate_l[2], err_rate_l[3]])

I_2 = np.log(I_Proj_2 / rate_2)
I_3 = np.log(I_Proj_3 / rate_3)
I_5 = np.log(I_leer / rate_5)

err_I_2 = np.sqrt((np.sqrt(rate_2) / rate_2) ** 2 + (err_I_Proj_2 / I_Proj_2) ** 2)
err_I_3 = np.sqrt((np.sqrt(rate_3) / rate_3) ** 2 + (err_I_Proj_3 / I_Proj_3) ** 2)
err_I_5 = np.sqrt((np.sqrt(rate_5) / rate_5) ** 2 + (err_I_leer / I_leer) ** 2)

print('''
    ###############################################################

    ~~~ Raten der verschiedenen Wuerfel ~~~

    Alu_leer: (Projektionen 2, 9, 10, 11)
    -----------------------------------------------
    Werte = {}
    Fehler = {}

    Alu_leer mit Mittelung: (alle 12 Projektionen)
    -----------------------------------------------
    Werte = {}
    Fehler = {}

    Wuerfel 2: (Projektionen 2, 9, 10, 11)
    -----------------------------------------------
    Werte = {}
    Fehler = {}

    Wuerfel 3: (Projektionen 2, 10, 11)
    -----------------------------------------------
    Werte = {}
    Fehler = {}

    Wuerfel 5: (alle {} Projektionen)
    -----------------------------------------------
    Werte = {}
    Fehler = {}

    ###############################################################
    '''
      .format(rate_l, err_rate_l, I_leer, err_I_leer, rate_2, err_rate_2,
              rate_3, err_rate_3, len(rate_5), rate_5, err_rate_5))

print('''
~~~ Logarithmen der Raten der verschiedenen Wuerfel ~~~

Wuerfel 2:
-----------------------------------------------
Werte = {}
Fehler = {}

Wuerfel 3:
-----------------------------------------------
Werte = {}
Fehler = {}

Wuerfel 5:
-----------------------------------------------
Werte = {}
Fehler = {}

###############################################################
'''.format(I_2, err_I_2, I_3, err_I_3, I_5, err_I_5)
      )

# Gewichtungsmatrizen
W_2 = np.diag(1 / err_I_2 ** 2)
W_3 = np.diag(1 / err_I_3 ** 2)
W_5 = np.diag(1 / err_I_5 ** 2)

# Über kleinsteQuadrate mu berechnen
mu_2, err_mu_2 = kleinsteQuadrate(I_2, W=W_2, A=A_2)
mu_3, err_mu_3 = kleinsteQuadrate(I_3, W=W_3, A=A_3)
mu_5, err_mu_5 = kleinsteQuadrate(I_5, W=W_5, A=A_4)

print(
    '''
~~~ Absorptionskoeffizienten der verschiedenen Wuerfel ~~~

Wuerfel 2:
-----------------------------------------------
Werte = {}
Fehler = {}

Wuerfel 3:
-----------------------------------------------
Werte = {}
Fehler = {}

Wuerfel 5:
-----------------------------------------------
Werte = {}
Fehler = {}
'''.format(mu_2, err_mu_2, mu_3, err_mu_3, mu_5, err_mu_5)
)

f1 = (0.203 - 0.23) / 0.203
f2 = (1.245 - 0.93) / 1.245
f3 = (0.203 - 0.26) / 0.203
f4 = (0.614 - 0.81) / 0.614
f5 = (0.203 - 0.15) / 0.203
f7 = (1.245 - 1.08) / 1.245
f6 = (0.203 - 0.23) / 0.203
f8 = (0.203 - 0.15) / 0.203
f9 = (0.116 - 0.01) / 0.116
f10 = (0.203 - 0.23) / 0.203
f11 = (0.116 - 0.05) / 0.116
print(f1)
print(f2)
print(f3)
print(f4)
print(f5)
print(f7)
print(f8)
print(f10)
print(f11)
