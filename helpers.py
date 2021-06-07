# helpers.py
# Fonctions pour l'examen intermédiaire 2021 de mécanique des Fluides pour GC
# Par Julien Marie F Ars

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MaxNLocator, MultipleLocator, AutoMinorLocator)

#Encodage des données en python
Q = 100e-3 #conversion L/s en m³/s
i=1e-2 #1%
W=50e-2 #conversion cm en m
K = 55 #m^(1/3)/s
b= 10e-2 #m
g = 9.81 #m / s^2
rho = 1e3 #kg / m^3


## Problème 1, ex 5

def H_star(x):
    return x + 1/(2*x**2)
        
def get_H_S(h, h_c):
    return h_c * H_star(h/h_c)

def plot_h_S(h_c, name = "", limits = (1,1), subplot = 111) :
    xi = np.linspace(0,8,1000)[1:]
    filter = np.bitwise_and(xi*h_c < limits[0], H_star(xi)*h_c < limits[1])
    
    plt.plot(xi[filter]*h_c, H_star(xi)[filter] * h_c, label=f"$H_S(h)$, {name}")
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.xaxis.set_major_formatter('{x:.2f}')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which="major", axis="y", alpha=0.8)
    plt.grid(which="minor", axis="y", alpha=0.2)
    
    plt.xlabel("$h$ (m)")
    plt.ylabel("$H_S$ (m)")
    plt.title(f"$H_S(h)$")

def plot_points(h, labels, h_c):
    h = np.array(h)
    H_S = get_H_S(h, h_c)
    print("".join([f"Point {name} : \th: {h*100:.0f} cm, Charge spécifique H_S : {H_S*100:.0f} cm\n" for name, h, H_S in zip(labels, h, H_S)]))
    
    ax = plt.gca()
    [ax.plot(h, H_S, "o", label=name) for name, h, H_S in zip(labels, h, H_S)]
    ax.legend()
    
# Problème 1, ex 6
def h1(x, var = globals()):
    h_A = var["h_A"]
    h_n_amont = var["h_n_amont"]
    x_ressaut = var["x_ressaut"]
    
    h_AB = h_A - i*x
    h_CD = h_n_amont
    try : #if numpy
        len(x)
        h = np.zeros_like(x)
        h[x<x_ressaut] = h_AB[x<x_ressaut]
        h[x>=x_ressaut] = h_CD
    except: #if not numpy
        if x<x_ressaut:
            h = h_AB
        else:
            h = h_CD
    return h

def draw_h(h = h1, figsize = (10, 5),var = globals()):
    x = np.linspace(0,100, 10000)
    
    plt.figure(figsize = figsize)
    plt.plot(x, x*i, label = "Fond")
    plt.plot(x, x*i+h(x, var), label = "Surface de l'eau")
    plt.plot(x, x*i+np.full_like(x, var["h_c_canal"]), "--", label = "Hauteur critique canal")
    plt.plot(x, x*i+np.full_like(x, var["h_c_fente"]), "--", label = "Hauteur critique fente")
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which="major", axis="both", alpha=0.8)
    plt.grid(which="minor", axis="y", alpha=0.2)
    
    plt.title("Représentation du problème")
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    
    plt.legend()
    plt.show()
    
    
    
## Problème 2
    
def draw_h2(x, h2, h1=h1, figsize = (10,5), var = globals()):
    plt.figure(figsize = figsize)
    plt.plot(x, x*i, label = "Fond")
    plt.plot(x, x*i+np.full_like(x, var["h_c_canal"]), "--", alpha = 0.5, label = "Hauteur critique canal")
    plt.plot(x, x*i+h1(x, var), label = "Surface de l'eau (problème 1)")
    plt.plot(x, x*i+h2, label = "Surface de l'eau (problème 2)")
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    plt.grid(which="major", axis="both", alpha=0.8)
    plt.grid(which="minor", axis="y", alpha=0.2)
    
    plt.xlabel('$x$ (m)')
    plt.ylabel('$y$ (m)')
    plt.title("Solution de la courbe de remous")
    
    plt.legend()
    plt.show()
    
    
    
    
## Problème 3, Question (6)
    
def draw_h3(var, x_h2):
    profil_hauteur_x = np.loadtxt("profil_hauteur_x.txt").T
    profil_hauteur_z = np.loadtxt("profil_hauteur_z.txt").T
    x = profil_hauteur_x[0]
    profil_x = profil_hauteur_x[1]
    z = profil_hauteur_z[0]
    profil_z = profil_hauteur_z[1]
    
    profil_z_theorique = np.full_like(z, var["h_A"]) # On considère h_A comme hauteur théorique
    #profil_z_theorique[np.bitwise_and(0.2 <z, z<0.3)] = h_c_fente
    
    #-- selon x
    
    plt.figure(figsize=(18,6))
    ax = plt.subplot(1,3,(1,2))
    
    plt.title("Profil de la hauteur selon x")
    plt.plot(x, i*x, label = "Fond")
    plt.plot(x, profil_x, label="Expérimental")
    plt.plot(x, h1(x, var)+i*x, label="Théorique - P1")
    plt.plot(x_h2, var["h2"]+i*x_h2, label = "Théorique - P2")
    plt.plot(x, var["h_c_canal"]+i*x,":", alpha=0.5, label = "Hauteur critique")
    
    plt.ylim([0,1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which="major", axis="y", alpha=0.8)
    plt.grid(which="minor", axis="y", alpha=0.2)
    
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.legend()
    
    #-- selon z
    
    ax = plt.subplot(1,3,3)
    plt.title("Profil de la hauteur selon z")
    plt.plot(z, np.zeros_like(z), label = "Fond")
    plt.plot(z, profil_z, label="Expérimental")
    plt.plot(z, profil_z_theorique, label="Théorique")
    plt.ylim([0,1])
    
    plt.ylim([0,1])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_formatter('{x:.2f}')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.grid(which="major", axis="both", alpha=0.8)
    plt.grid(which="minor", axis="y", alpha=0.2)
    ax.yaxis.tick_right()
    
    plt.xlabel("$z$ (m)")
    plt.legend()
    plt.tight_layout()
    plt.show()