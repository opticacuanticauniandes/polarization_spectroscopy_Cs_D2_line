# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 14:38:43 2025

@author: jr.mejia1228
"""

# -*- coding: utf-8 -*-
"""
Optimized OBE solver for Cesium transitions with magnetic field
"""

import numpy as np
from scipy.special import erf
from sympy.physics.wigner import wigner_3j, wigner_6j
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate
from numpy.linalg import eig, solve
import time
from functools import lru_cache
import pandas as pd

# Start timing the execution
start_time = time.time()

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
a = 800  # Radius of beam in micrometers
Gamma = 5.1  # MHz*rad
Ipump = 1  # Intensity of the pump with Isat=1
Isat = 1
gJground = 2.00254032  # Landé g factor for 6^2Subscript[S, 1/2]
gJexcited = 1.3340  # Landé g factor for 6^2Subscript[P, 3/2]
muB = 9.27400968e-24  # Bohr magneton in J T^-1
h = 6.62607015e-34  # Planck constant in J s
gammaground = gJground * muB / h
gammaexcited = gJexcited * muB / h
B = 1e-7  # Magnetic field

# Most probable speed in m/s
u = np.sqrt(2 * 1.380649e-23 * T / m)

# Angular momenta {Le_, Lg_, Je_, Jg_, S_, I_}
AM = [1, 0, 3/2, 1/2, 1/2, 7/2]

# Precompute constants to avoid recalculation
const_2a_u = (2 * a) / u
print("Most probable speed (u):", u)
print("2a/u:", const_2a_u)

# Cache functions for repeated calculations
@lru_cache(maxsize=1024)
def wigner_3j_cached(j1, j2, j3, m1, m2, m3):
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))

@lru_cache(maxsize=1024)
def wigner_6j_cached(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))

# Function Definitions with caching
@lru_cache(maxsize=1024)
def R(Fe, me, Fg, mg, Le, Lg, Je, Jg, S, IN):
    """Normalized Line Strength with caching for repeated values"""
    w6j_1 = wigner_6j_cached(Le, Je, S, Jg, Lg, 1)
    w6j_2 = wigner_6j_cached(Je, Fe, IN, Fg, Jg, 1)
    w3j = wigner_3j_cached(Fg, 1, Fe, mg, me-mg, -me)
    
    return (2 * Le + 1) * (2 * Je + 1) * (2 * Jg + 1) * (2 * Fe + 1) * (2 * Fg + 1) * (w6j_1 * w6j_2 * w3j) ** 2

def H(t):
    """H function - optimized by vectorizing and using constants"""
    x = const_2a_u / t
    return (1 / t) * (-1 + np.sqrt(np.pi) / (2 * x) * (1 + 2 * x**2) *
                      np.exp(-(x)**2) * erf(1j * x) / (1j))

@lru_cache(maxsize=1024)
def gF(gJ, F, J, IN):
    """Landé Factor for the Hyperfine Structure, including Bohr's magneton and Planck Constant"""
    return gJ * ((F * (F + 1) + J * (J + 1) - IN * (IN + 1)) / (2 * F * (F + 1))) * muB / h

# Precompute gF values for common combinations
gF_cache = {}
for Fe_val in range(F-1, F+2):
    gF_cache[('excited', Fe_val)] = gF(gJexcited, Fe_val, AM[2], AM[5])
for Fg_val in [F, Fp]:
    gF_cache[('ground', Fg_val)] = gF(gJground, Fg_val, AM[3], AM[5])

@lru_cache(maxsize=1024)
def Delta(Fg, Fe, mg, me, B_field, Le, Lg, Je, Jg, S, IN):
    """Frequency differences with caching"""
    # Base frequency differences
    base_delta = {
        (5, 4): 251.00,
        (4, 3): 201.24,
        (3, 2): 151.21,
        (5, 3): 251.00 + 201.24,
        (5, 5): 0
    }
    
    # Get base value
    base_value = base_delta.get((Fg, Fe), 0)
    
    # Add Zeeman shift
    zeeman_shift = (gF_cache.get(('excited', Fe), 0) * me - 
                    gF_cache.get(('ground', Fg), 0) * mg) * B_field * 1e-6
    
    return base_value + zeeman_shift

# Plot H(t) function
tmin=0.32 #Time in miliseconds to plot the H(t) function
tmax=50
t_values = np.linspace(tmin,tmax, 1000)
H_values = np.real(np.vectorize(H)(t_values))

plt.figure(figsize=(10, 6))
plt.plot(t_values, H_values, label="H(t)")
plt.xlabel(r"$t [\mu s]$", fontsize=14)
plt.ylabel(r"$H(t)$", fontsize=14)
plt.title("Probability density of interaction times between atoms and laser", fontsize=16)
plt.grid(False)
plt.show()

# Numerical Integration of H(t)
integral_result, error = integrate.quad(lambda t: np.real(H(t)), 0.32, 50)
print(f"Numerical Integration of H(t) from t={tmin}ms to t={tmax}ms:", integral_result)

# Create symbolic variables for rate equations
P = {}
Q = {}
t = sp.symbols('t')
for Fe in range(3, 6):
    for m in range(-6, 7):
        P[(Fe, m)] = sp.Function(f'P_{Fe}_{m}')(t)
        Q[(Fe, m)] = sp.Function(f'Q_{Fe}_{m}')(t)

# Define variables list only once
variables = []
labels=[]
for Fg in range(Fp, F+1):
    for mg in range(-Fg, Fg+1):
        variables.append(P[(Fg, mg)])  # Ground states
        label = f"P({Fg},{mg})"
        labels.append(label)  # Store label for each plot
for Fe in range(F-1, F+2):
    for me in range(-Fe, Fe+1):
        variables.append(Q[(Fe, me)])  # Excited states
        label = f"Q({Fe},{me})"
        labels.append(label)  # Store label for each plot

# Precompute Delta values for all possible combinations
delta_cache = {}
for q_val1 in [-1,0,1]:
    for Fe_val in range(F-1, F+2):
        for mg in range(-F, F+1):
            for me in range(-Fe_val, Fe_val+1):
                key = (F+1, Fe_val, mg+q_val1, me)
                delta_cache[key] = Delta(F+1, Fe_val, mg+q_val1, me, B, *AM)

# Precompute R values for common combinations
r_cache = {}
for q_val1 in [-1,0,1]:
    for Fe_val in range(F-1, F+2):
        for Fg in [F, Fp]:
            for me in range(-Fe_val, Fe_val+1):
                for mg in range(-Fg, Fg+1):
                    key = (Fe_val, me, Fg, mg+q_val1)
                    r_cache[key] = R(Fe_val, me, Fg, mg+q_val1, *AM)

def Ground1M(Delta_p, q):
    """Ground state equations for F=4"""
    ground1_eqs = []
    for mg in range(-F, F+1):
        sum_Fe = 0
        for Fe in range(F-1, F+2):  # Fe = F-1, F, F+1
            # First term
            r_val = r_cache.get((Fe, mg+q, F, mg), 0)
            delta_val = delta_cache.get((F+1, Fe, mg, mg+q), 0)
            term1 = (
                r_val * Gamma / 2 * (Ipump / Isat) * 
                (Q[(Fe, mg+q)] - P[(F, mg)]) /
                (1 + 4 * ((Delta_p + delta_val)**2) / Gamma**2)
            )
            
            # Second term
            sum_me = 0
            for i in range(3):  # me = mg-1, mg, mg+1
                me = mg - 1 + i
                if -Fe <= me <= Fe:  # Check bounds
                    r_val = r_cache.get((Fe, me, F, mg), 0)
                    sum_me += Gamma * r_val * Q[(Fe, me)]
            
            sum_Fe += term1 + sum_me
        ground1_eqs.append(sum_Fe)
    return ground1_eqs

def Ground2M(Delta_p, q):
    """Ground state equations for F=3"""
    eqs = []
    for mg in range(-Fp, Fp+1):
        eq = 0
        for Fe in range(Fp, Fp+2):
            term = 0
            for me in range(mg-1, mg+2):
                if -Fe <= me <= Fe:  # Check bounds
                    if (Fe, me) in Q:
                        r_val = r_cache.get((Fe, me, Fp, mg), 0)
                        eq += Gamma * r_val * Q[(Fe, me)]
            eq += term
        eqs.append(eq)
    return eqs

def Excited1M(Delta_p, q):
    """Excited state equations for Fe=3"""
    eqs = []
    for me in range(-(F-1), F):
        eq = 0
        # First term
        r_val = r_cache.get((F-1, me, F, me-q), 0)
        delta_val = delta_cache.get((F+1, F-1, me-q, me), 0)
        first_term = -r_val * Gamma/2 * (Ipump/Isat) * \
                     (Q[(F-1, me)] - P[(F, me-q)]) / \
                     (1 + 4 * ((Delta_p + delta_val)**2) / Gamma**2)
        
        # Second term
        second_term = 0
        for Fg in range(Fp, F+1):
            for mg in range(me-1, me+2):
                if -Fg <= mg <= Fg:  # Check bounds
                    r_val = r_cache.get((F-1, me, Fg, mg), 0)
                    second_term += Gamma * r_val * Q[(F-1, me)]
        
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

def Excited2M(Delta_p, q):
    """Excited state equations for Fe=4"""
    eqs = []
    for me in range(-(F), F+1):
        eq = 0
        
        # First term
        r_val = r_cache.get((F, me, F, me-q), 0)
        delta_val = delta_cache.get((F+1, F, me-q, me), 0)
        first_term = -r_val * Gamma/2 * (Ipump/Isat) * \
                     (Q[(F, me)] - P[(F, me-q)]) / \
                     (1 + 4 * ((Delta_p + delta_val)**2) / Gamma**2)
        
        # Second term
        second_term = 0
        for Fg in range(Fp, F+1):
            for mg in range(me-1, me+2):
                if -Fg <= mg <= Fg:  # Check bounds
                    r_val = r_cache.get((F, me, Fg, mg), 0)
                    second_term += Gamma * r_val * Q[(F, me)]
        
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

def Excited3M(Delta_p, q):
    """Excited state equations for Fe=5"""
    eqs = []
    for me in range(-(F+1), F+2):
        eq = 0
        
        # First term
        r_val = r_cache.get((F+1, me, F, me-q), 0)
        delta_val = delta_cache.get((F+1, F+1, me-q, me), 0)
        first_term = -r_val * Gamma/2 * (Ipump/Isat) * \
                     (Q[(F+1, me)] - P[(F, me-q)]) / \
                     (1 + 4 * ((Delta_p + delta_val)**2) / Gamma**2)
        
        # Second term
        second_term = 0
        for Fg in range(Fp, F+1):
            for mg in range(me-1, me+2):
                if -Fg <= mg <= Fg:  # Check bounds
                    r_val = r_cache.get((F+1, me, Fg, mg), 0)
                    second_term += Gamma * r_val * Q[(F+1, me)]
        
        eq += first_term - second_term
        eqs.append(eq)
    return eqs

# Precompute the set of rate equations once
def build_matrix(Delta_p, q):
    """Build the matrix of rate equations for a specific Delta_p and q value"""
    dydt = []
    # Ground state equations
    dydt.extend(Ground2M(Delta_p, q))
    dydt.extend(Ground1M(Delta_p, q))
    # Excited state equations
    dydt.extend(Excited1M(Delta_p, q))
    dydt.extend(Excited2M(Delta_p, q))
    dydt.extend(Excited3M(Delta_p, q))
    
    matrix = sp.linear_eq_to_matrix(dydt, variables)
    matrix2work = matrix[0]
    matrixnum = np.array(matrix2work).astype(np.float64)
    return matrixnum


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


# Define a vectorized version of H for faster integration
def H_vec(t_array):
    return np.array([np.real(H(t_val)) for t_val in t_array])

# Solve the rate equations with improved approach
def rate_equations(Delta_p_array, q, t_array):
    """Solve the rate equations for multiple Delta_p values"""
    results = []
    initial_conds = np.concatenate((np.full(16, 1/16), np.zeros(43 - 16)))
    
    # Show progress
    total = len(Delta_p_array)
    checkpoint = max(1, total // 20)  # Update every 5%
    
    for idx, delta_val in enumerate(Delta_p_array):
        if idx % checkpoint == 0:
            print(f"Processing {idx}/{total} delta values ({idx/total*100:.1f}%)")
        
        # Build and solve the matrix for this delta value
        matrix = build_matrix(delta_val, q)
        U,S,Vh=np.linalg.svd(matrix,full_matrices=False)
        matrix2=U @ np.diag(S) @ Vh
        w, v = eig(matrix2)
        coefs = solve(v, initial_conds)
        
        # Calculate solution for all time points
        y = []
        sol=[]
        for t_val in t_array:
            y.append(np.real(H(t_val)) * v @ (coefs * np.exp(w * t_val)))
            sol.append( v @ (coefs * np.exp(w * t_val)))
        # Integrate over all time points
        integrando = np.transpose(y)
        integral_res = integrate.simpson(integrando, x=t_array)
        results.append(np.real(integral_res))
    
    return [results,y,sol]

def calculate_anisotropy(Delta_p_array, populations, delta_array):
    """Calculate anisotropy for a range of delta values with optimized performance"""
    pops = np.transpose(populations)
    
    # Create dictionaries with population values
    U_plus = {}
    V_plus = {}
    
    # Ground state populations (U)
    idx = 0
    for Fg, mg_range in [(3, range(-3, 4)), (4, range(-4, 5))]:
        for mg in mg_range:
            U_plus[(Fg, mg)] = pops[idx]
            idx += 1
    
    # Add zeros for indices not in pops
    for Fg in [3, 4]:
        for mg in range(-5, 6):
            if (Fg, mg) not in U_plus:
                U_plus[(Fg, mg)] = np.zeros(len(Delta_p_array))
    
    # Excited state populations (V)
    for Fe, me_range in [(3, range(-3, 4)), (4, range(-4, 5)), (5, range(-5, 6))]:
        for me in me_range:
            V_plus[(Fe, me)] = pops[idx] if idx < len(pops) else np.zeros(len(Delta_p_array))
            idx += 1
    
    # Add zeros for indices not in pops
    for Fe in [3, 4, 5]:
        for me in range(-5, 6):
            if (Fe, me) not in V_plus:
                V_plus[(Fe, me)] = np.zeros(len(Delta_p_array))
    
    q_values = [-1, 1]
    anisotropy = []
    
    # Show progress
    total = len(delta_array)
    checkpoint = max(1, total // 20)  # Update every 5%
    
    # Precompute gaussian factors for all Delta_p and delta combinations
    gauss_factors = {}
    for i, delta_i in enumerate(delta_array):
        for j, delta_p in enumerate(Delta_p_array):
            gauss_factors[(i,j)] = np.exp(-(delta_p - delta_i)**2 / ((k * u * 1e-6)**2))
    
    for i, delta_i in enumerate(delta_array):
        if i % checkpoint == 0:
            print(f"Calculating anisotropy {i}/{total} ({i/total*100:.1f}%)")
        
        integrand = []
        for j, delta_p in enumerate(Delta_p_array):
            term2 = 0
            for Fe_val in range(F-1, F+2):
                for mg in range(-F, F+1):
                    for q_val in q_values:
                        me = mg + q_val
                        if -Fe_val <= me <= Fe_val:  # Check bounds
                            r_val = r_cache.get((Fe_val, me, F, mg), 0)
                            delta_val = delta_cache.get((F+1, Fe_val, mg, me), 0)
                            
                            denominator = 1 + 4 * ((2 * delta_i - delta_p + delta_val) ** 2) / (Gamma ** 2)
                            numerator = r_val * q_val * (U_plus[(F, mg)][j] - V_plus[(Fe_val, me)][j]) * (
                                (2 * delta_i - delta_p + delta_val) / Gamma
                            )
                            term2 += numerator / denominator
            
            integrand.append(term2 * gauss_factors[(i,j)])
        
        anisotropy.append(integrate.simpson(integrand, x=Delta_p_array))
    
    return anisotropy

# Main execution
def main():
    # Parameters for calculation
    num_datos = 1400 # Reduced from 2800 for better performance
    t_array = np.linspace(tmin, tmax, 100)  # Reduced from 100
    Delta_p_array = np.linspace(-550,150, num_datos)
    q_pump = -1
    
    print(f"Starting population calculations with {num_datos} points...")
    # Calculate populations
    populations_plus = rate_equations(Delta_p_array, q_pump, t_array)
    
    

    
    
    #Plot solutions to optical Bloch equations
    plt.plot(t_array, populations_plus[2],label=labels)
    plt.xlabel(r"$t [\mu s]$", fontsize=14)
    plt.ylabel('Relative populations')
    plt.title('Populations')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=10)
    plt.show()
    
    
    #Plot solutions to optical Bloch equations weighted by function H
    plt.plot(t_array, populations_plus[1],label=labels)
    plt.xlabel(r"$t [\mu s]$", fontsize=14)
    plt.ylabel('Relative populations *H(t)')
    plt.title('Populations weighted by interaction time')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=10)
    plt.show()
    
    # Plot populations
    plt.figure(figsize=(10, 6))
    plt.plot(Delta_p_array, populations_plus[0])  # Plot excited states
    plt.xlabel(r"$ \delta pump[MHz]$", fontsize=14)
    plt.ylabel('Relative populations')
    plt.title('Excited State Populations')
    plt.show()
    
    # Calculate anisotropy
    print("Starting anisotropy calculations...")
    delta_array = np.linspace(-550, 150, num_datos)
    anisotropy = calculate_anisotropy(Delta_p_array, populations_plus[0], delta_array)
    
    # Plot anisotropy
    plt.figure(figsize=(10, 6))
    plt.plot(delta_array, anisotropy)
    plt.xlabel(r"$ \delta probe[MHz]$", fontsize=14)
    plt.ylabel(r"$\Delta n$")
    plt.title('Anisotropy')
    plt.show()
    
    # Save results
    filename = f'B{B}_T{T}_a{a}_I_{Ipump}_q{q_pump}_numdatos{num_datos}_optimized.csv'
    df = pd.DataFrame({'detuning[MHz]': delta_array, 'Anisotropy': anisotropy})
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    # Print execution time
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()