# helpers.py
# Fonctions pour l'examen intermédiaire 2021 de mécanique des Fluides pour GC
# Par Julien Marie F Ars

import numpy as np
import matplotlib.pyplot as plt

## Problème 1, ex 5

def H_star(x):
    return x + 1/(2*x**2)
        
def get_H_S(h, h_c):
    return h_c * H_star(h/h_c)

def plot_h_S(h_c, name = "", limits = (1,1), subplot = 111) :
    xi = np.linspace(0,8,1000)[1:]
    filter = np.bitwise_and(xi*h_c < limits[0], H_star(xi)*h_c < limits[1])
    
    plt.subplot(subplot)
    plt.plot(xi[filter]*h_c, H_star(xi)[filter] * h_c, label=f"$H_S(h)$, {name}")
    
    plt.xlabel("$h$")
    plt.ylabel("$H_S$")
    plt.title(f"$H_S(h)$, {name}")

def plot_points(h, labels, h_c):
    h = np.array(h)
    H_S = get_H_S(h, h_c)
    print("".join([f"Point {name} : \th: {h*100:.0f} cm, Charge spécifique H_S : {H_S*100:.0f} cm\n" for name, h, H_S in zip(labels, h, H_S)]))
    
    ax = plt.gca()
    [ax.plot(h, H_S, "o", label=name) for name, h, H_S in zip(labels, h, H_S)]
    ax.legend()
    
"""
x= np.array([h_B, h_n_amont])
y= get_H_S(x, h_c_canal)
labels = ["B","C et D"]
plt.plot(x,y,"bo")
ax = plt.gca()
[ax.annotate(labels[i], (x[i]+0.005, y[i]+0.005)) for i in range(len(x))]

## Fente
#  -----
plt.subplot(122)
plt.plot(xi[filter(h_c_fente)]*h_c_fente, H_star[filter(h_c_fente)] * h_c_fente)
plt.title("$H_S(h)$, fente")
# Point : A
x = np.array([h_A])
y = get_H_S(x, h_c_fente)
labels = ["A"]
plt.plot(x,y,"bo")
ax = plt.gca()
[ax.annotate(labels[i], (x[i]+0.01, y[i]+0.01)) for i in range(len(x))]

plt.show()


ax = plt.gca()
ax.plot(h, h_S, "ro")
marge = 1/20
[ax.annotate(labels[i], (h[i], h_S[i]+i * marge)) for i in range(len(h))]

"""
    