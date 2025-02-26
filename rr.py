import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.optimize import curve_fit
import csv
from scipy.signal import find_peaks
import statistics
import pandas as pd

###intensity ratio means transmission

def transmission_model(w, a, r):
    """
    ###refer to calculate transmission for parameters
    this is only the equation of the transmission, no input
    serves as a model for the curve fit
    """
    radius = 150e-6
    n_eff = 1.8
    L = (2 * np.pi) * radius  # Circumference in meters
    beta = (2 * np.pi * n_eff) / w # propogation constant (radians per meter)
    phi = beta * L  # Phase shift in radians
    numerator = a**2 - (2 * r * a * np.cos(phi)) + r**2
    denominator = 1 - (2 * a * r * np.cos(phi)) + (r * a)**2  
    return numerator / denominator

def calculate_transmission(a=0.95, r=0.95, n_eff=1.8, radius=150e-6, wavelength_range=(560e-9, 565e-9), num_points=10000):
    """
    calculates transmission for a range of wavelengths based on given parameters
    
    parameters:
        a : single pass amplitude transmission ### affects amplitude
        r : self-coupling coefficient ### affects amplitufe
        n_eff : effective refractive index
        radius : ring radius in meters ### affects period of wave
        wavelength_range : min and max wavelengths in meters
        num_points : Number of points in the wavelength range

    returns:
        wavelengths (numpy array): array of wavelength values
        intensity_ratios (list): corresponding transmission with noise.
    """
    L = (2 * np.pi) * radius  # round trip length in meters
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], num_points)
    intensity_ratios = []
    
    for w in wavelengths:
        beta = (2 * np.pi * n_eff) / w  # propogation constant (radians per meter)
        phi = beta * L  # phase shift in radians
        numerator = a**2 - (2 * r * a * np.cos(phi)) + r**2
        denominator = 1 - (2 * a * r * np.cos(phi)) + (r * a)**2  
        transmission = numerator / denominator 

        # generate random noise
        rand_num = random.uniform(0, transmission * 0.05) 
        transmission += rand_num if random.choice([True, False]) else -rand_num
        
        intensity_ratios.append(transmission)

    
    return wavelengths, intensity_ratios

def create_csv(filename, wavel, transm):
    with open(filename, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(wavel)

        writer.writerow(transm)



def fit_transmission(wavelengths, intensity_ratios):
    """
    -curve fit
    -takes in list of wavelenghts and its corresponding transmission
    -for p0, use random values unless they are known
    """
    
    # Fit the model to the data
    popt, _ = curve_fit(transmission_model, wavelengths, intensity_ratios, p0=[0.90,0.70])
    a = popt[0]
    r = popt[1]
    
    return a, r

#def calculate_loss():

def fwhm(a, r, min_wavelengths):
    radius = 150e-6
    L = (2 * np.pi) * radius  # Circumference in meters
    n_g = 1.5
    fwhm_list=[]

    for wave in min_wavelengths:
        numerator = (1- r * a) * wave **2
        denominator = (np.pi * n_g * L) * np.sqrt(r * a)
        fullwhm = numerator/denominator
        fwhm_list.append(float(fullwhm))

    return fwhm_list

def q_loaded(min_wavelengths, fwhm):
    loaded_list = [min_wavelengths[i] / fwhm[i] for i in range(len(min_wavelengths))]
    return loaded_list
        
def q_internal(loaded, t):
    internal_list = [(2 * loaded[i])/ (1 + np.sqrt(t[i])) for i in range(len(loaded))]
    return internal_list

def loss(q_int, wavelength):
    n_g = 1.5
    loss_list = [(2 * np.pi * n_g) / (q_int[i] * wavelength[i]* 1e2 ) for i in range(len(wavelength))]
    return loss_list

def plot_results(wavelengths, intensity_ratios, a, r, widths, internal_factors, loaded_factors, loss_values):
    """
    Plots the original noisy data and the best fit curve.
    """
    fitted_intensities = transmission_model(wavelengths, a, r)
    minima_wavelengths, minima_transmissions = find_minima(wavelengths, intensity_ratios)

    avg_fwhm = statistics.mean(widths)
    avg_q_int = statistics.mean(internal_factors)
    avg_q_loaded = statistics.mean(loaded_factors)
    avg_loss = statistics.mean(loss_values)
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths * 1e9, intensity_ratios, label='Noisy Data', color='b', alpha=0.5)
    plt.plot(wavelengths * 1e9, fitted_intensities, label= f'Best Fit: a={a}, r={r}', color='r', linewidth=1)
    plt.scatter(minima_wavelengths * 1e9, minima_transmissions, color='black', 
                label=(f'Minima. Avg FWHM: {avg_fwhm:.3e}, '
                   f'Avg Q (Int): {avg_q_int:.3f}, '
                   f'Avg Q (Loaded): {avg_q_loaded:.3f}, '
                   f'Avg Loss: {avg_loss:.3f}'),
                     s=10, zorder=3)


    plt.legend(loc = 'upper right')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title('Transmission vs Wavelength with Best Fit')
    plt.legend()
    plt.grid()
    plt.show()


def find_minima(wavelenghts, intensity_ratios):
    """finds the minima wavelenghts"""
    opposite_transm = -np.array(intensity_ratios)
    peaks, _ = find_peaks(opposite_transm, distance=50, height = -0.2)

    minima_waves = wavelenghts[peaks]
    minima_transmissions = np.array(intensity_ratios)[peaks]
    return minima_waves, minima_transmissions

"""
    ###uncomment if you want the graph with minimum peaks only###
def plot_minima(minima_wavelengths, minima_transmissions):
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths * 1e9, intensities, label='Noisy Data', color='b', alpha=0.5)
    plt.scatter(minima_wavelengths * 1e9, minima_transmissions, color='red', label='Minima', s=10, zorder=3)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title('Transmission vs Wavelength with Minima')
    plt.legend()
    plt.grid()
    plt.show()
    """

# Generate noisy transmission data
wavelengths, intensities = calculate_transmission()

# Save fake data to csv file
filename = 'rr_test.csv'
create_csv(filename, wavelengths, intensities)

# Fit the model to the data
a, r = fit_transmission(wavelengths, intensities)
print(f"Optimal values: a={a}, r={r}")

#Find FWHM



##FIND PEAK
minima_wavelenghts, minima_transmissions = find_minima(wavelengths, intensities)
print(f"Wavelengths: {minima_wavelenghts}")
print(f"Transmission: {minima_transmissions}")

widths = fwhm(a, r, minima_wavelenghts)
print(f"FWHM: {widths}")

#graph with minima plotted
#plot_minima(minima_wavelenghts, minima_transmissions)

loaded_factors = list(map(float, q_loaded(minima_wavelenghts, widths)))
print(f"Q-Loaded: {loaded_factors}")

internal_factors = list(map(float, q_internal(loaded_factors, minima_transmissions)))
print(f"Q-Internal: {internal_factors}")

loss_values = list(map(float, loss(internal_factors, minima_wavelenghts)))
print(f"Loss: {loss_values}") 


##make graph with average FWHM, Qint, Qloaded, Loss
f_avg = statistics.mean(widths)
print(f_avg)

plot_results(wavelengths, intensities, a, r, widths, internal_factors, loaded_factors, loss_values)


df = pd.read_csv("rr_test.csv", header= None)
print(df.shape)
x = df.iloc[0, :].values
#print(x)
y = df.iloc[1, :].values
print(y)

plt.plot(x,y)
plt.show()