 # -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 18:54:33 2025

@author: Jose Mejía
"""

import numpy as np
from scipy.special import erf
from sympy.physics.wigner import wigner_3j
from sympy.physics.wigner import wigner_6j
from sympy import symbols, Function, Sum, IndexedBase, Eq, diff, Derivative
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import sympy as sp
from sympy import shape
from numpy.linalg import eig,solve
from scipy.integrate import quad

# Constants and Variables
F = 4  # Transition the laser is tuned to
Fp = 3  # The other ground state
Fe = [F - 1, F, F + 1]  # Possible Excited states

# Frequency differences (in MHz)
Δ54 = 251.00
Δ43 = 201.24
Δ32 = 151.21
Δ53 = Δ54 + Δ43
Δ55 = 0  # Everything in MHz

# Physical Constants
T = 300  # Temperature in Kelvin
m = 2.207023979e-25  # Mass of Cs atoms in kg
Lambda = 852.34727582e-9  # Wavelength in meters
k = 2 * np.pi / Lambda  # Wave number
deNs = 1e-10  # Atomic density
a = 800 # Radius of beam in micrometers
Gamma = 5.1  # MHz*rad
Ipump = 0.5 # Intensity of the pump with Isat=1
Isat = 1

# Most probable speed in m/s
u = np.sqrt(2 * 1.380649e-23 * T / m)

# Angular momenta {Le_, Lg_, Je_, Jg_, S_, I_}
AM = [1, 0, 3/2, 1/2, 1/2, 7/2]

# Function Definitions

# Normalized Line Strength
def R(Fe, me, Fg, mg, Le, Lg, Je, Jg, S, IN):
    return (2 * Le + 1) * (2 * Je + 1) * (2 * Jg + 1) * (2 * Fe + 1) * (2 * Fg + 1) * (wigner_6j(Le, Je, S, Jg, Lg, 1) *
            wigner_6j(Je, Fe, IN, Fg, Jg, 1) *
            wigner_3j(Fg, 1, Fe, mg , me-mg , -me)) ** 2

# H function
def H(t):
    x = (2 * a) / (u * t)
    return (1 / t) * (-1 + np.sqrt(np.pi) / (2 * x) * (1 + 2 * x**2) *
                      np.exp(-(x)**2) * erf(1j * x) / (1j))


print("Most probable speed (u):", u)
t_values = np.linspace(0.5, 50, 1000)
H_values = np.real(H(t_values))

plt.figure(figsize=(10, 6))
plt.plot(t_values, H_values, label="H(t)")
plt.xlabel(r"$t [\mu s]$", fontsize=14)
plt.ylabel(r"$H(t)$", fontsize=14)
plt.title("Probability density of interaction times between atoms and laser", fontsize=16)
plt.grid(False)
plt.show()

# Numerical Integration
integral_result, error = integrate.quad(lambda t: np.real(H(t)),0.32, 150)
print("Numerical Integration of H(t) from t=0.32 to t=150:", integral_result)

# Calculating 2a/u
result_2a_u = (2 * a) / u
print("2a/u:", result_2a_u)

## Solving the Rate equations that comes from the OBE

# Define dictionaries for P and Q with negative indices
P = {}
Q = {}
t = sp.symbols('t')
for Fe in range(3, 6):
    for m in range(-6, 7):
        P[(Fe, m)] = sp.Function(f'P_{Fe}_{m}')(t)
        Q[(Fe, m)] = sp.Function(f'Q_{Fe}_{m}')(t)
AM = [1, 0, 3/2, 1/2, 1/2, 7/2]
  # Angular momenta: [Le, Lg, Je, Jg, S, I]
Delta = {
    (5, 4): 251.00,
    (4, 3): 201.24,
    (3, 2): 151.21,
    (5, 3): 251.00 + 201.24,
    (5, 5): 0  # Everything in MHz
}
def Ground1M(Delta_p, q):
    ground1_eqs = []
    for mg in range(-F, F+1):
        sum_Fe = 0
        for Fe in range(F-1, F+2):  # Fe = F-1, F, F+1
            term1 = (
                R(Fe, mg+q, F, mg, *AM) * Gamma / 2 * (Ipump / Isat) * 
                (Q[(Fe, mg+q)] - P[(F, mg)]) /
                (1 + 4 * ((Delta_p + Delta[F+1, Fe])**2) / Gamma**2)
            )
            sum_me = 0
            for i in range(3):  # me = mg-1, mg, mg+1
                me = mg - 1 + i
                sum_me += Gamma * R(Fe, me, F, mg, *AM) * Q[(Fe, me)]
            sum_Fe += term1 + sum_me
        ground1_eqs.append(sum_Fe)
    return ground1_eqs

def Ground2M(Delta_p, q):
    eqs = []
    for mg in range(-Fp, Fp+1):
        eq=0
        for Fe in range(Fp, Fp+2):
            term = 0
            for me in range(mg-1, mg+2):
                if (Fe, me) in Q:  # Check for valid indices
                    term += Gamma * R(Fe, me, Fp, mg, *AM) * Q[(Fe, me)]
            eq += term
        eqs.append(eq)
    return eqs

# Excited State Equations
def Excited1M(Delta_p, q):
    eqs = []
    for me in range(-(F-1), F):
        eq=0
        first_term = 0
        first_term = -R(F-1, me, F, me-q, *AM) * Gamma/2 * (Ipump/Isat) * \
                         (Q[(F-1, me)] - P[(F, me-q)]) / \
                         (1 + 4 * ((Delta_p + Delta[(F+1, F-1)])**2) / Gamma**2)
        second_term = 0
        for Fg in range(Fp, F+1): 
            for mg in range(me-1, me+2):
               second_term += Gamma * R(F-1, me, Fg, mg, *AM) * Q[(F-1, me)]
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

def Excited2M(Delta_p, q):
    eqs = []
    for me in range(-(F), F+1):
        eq=0
        first_term = 0
        first_term = -R(F, me, F, me-q, *AM) * Gamma/2 * (Ipump/Isat) * \
                         (Q[(F, me)] - P[(F, me-q)]) / \
                         (1 + 4 * ((Delta_p + Delta[(F+1, F)])**2) / Gamma**2)
        second_term = 0
        for Fg in range(Fp, F+1):  
            for mg in range(me-1, me+2):
                second_term += Gamma * R(F, me, Fg, mg, *AM) * Q[(F, me)]
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

def Excited3M(Delta_p, q):
    eqs = []
    for me in range(-(F+1), F+2):
        eq=0
        first_term = 0
        first_term = -R(F+1, me, F, me-q, *AM) * Gamma/2 * (Ipump/Isat) * \
                         (Q[(F+1, me)] - P[(F, me-q)]) / \
                         (1 + 4 * ((Delta_p + Delta[(F+1, F+1)])**2) / Gamma**2)
        second_term = 0
        for Fg in range(Fp, F+1):
            for mg in range(me-1, me+2):
                second_term += Gamma * R(F+1, me, Fg, mg, *AM) * Q[(F+1, me)]
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

variables=[]
for Fg in range(Fp, F+1):
    for mg in range(-Fg, Fg+1):
        variables.append(P[(Fg,mg)])  # Ground states
for Fe in range(F-1, F+2):
    for me in range(-Fe, Fe+1):
        variables.append(Q[(Fe,me)])     # Excited states     # Excited states start at 0


#matrix1=sp.linear_eq_to_matrix(Ground1M(0,1),variables)
#print(variables)
         

# Define the ODE system for the ground and excited states
def rate_equations(Delta_p,q):
    dydt = []
    # Ground state equations (Ground1M & Ground2M)
    dydt.extend(Ground2M(Delta_p, q))
    dydt.extend(Ground1M(Delta_p, q))
    # Excited state equations (Excited1M & Excited2M & Excited3M)
    dydt.extend(Excited1M(Delta_p, q))
    dydt.extend(Excited2M(Delta_p, q))
    dydt.extend(Excited3M(Delta_p, q))
    
    matrix=sp.linear_eq_to_matrix(dydt,variables)
    matrix2work=matrix[0]
    matrixnum=np.array(matrix2work).astype(np.float64)
    w,v=eig(matrixnum)
    initial_conds=np.concatenate((np.full(16, 1/16), np.zeros(43 - 16)))
    coefs=solve(v,initial_conds)

    return [coefs,w,np.transpose(v),matrixnum]

def solution(coefs,eigenvalues,eigenvects,t):
    y=[]
    for t_val in t:
        y.append(np.transpose(eigenvects)@(coefs*np.exp(eigenvalues*t_val)))
    return y

def integrand(t,coefs,eigenvalues,eigenvects):
    y=[]
    for t_val in t:
        y.append(np.real(H(t_val))*np.transpose(eigenvects)@(coefs*np.exp(eigenvalues*t_val)))
    return np.transpose(y)


Delta_p=50
q=-1
t = np.linspace(0.32, 50, 100)
sol=rate_equations(Delta_p,q)
rates=solution(sol[0],sol[1],sol[2],t)
plt.plot(t, rates)
plt.xlabel(r"$t [\mu s]$", fontsize=14)
plt.ylabel('Relative populations')
plt.title('Populations')
plt.show()

# Ahora la integral numérica
t = np.linspace(0.32, 50, 100)
integrando=integrand(t,sol[0],sol[1],sol[2])
plt.plot(t, np.transpose(integrando))
plt.xlabel(r"$t [\mu s]$", fontsize=14)
plt.ylabel(r"$Relative populations*H(t)$")
plt.title('Weighted Populations')
plt.show()
I1 = integrate.simpson(integrando,x=t)
print(I1)

