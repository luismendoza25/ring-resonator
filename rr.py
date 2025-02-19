import numpy as np
from matplotlib import pyplot as plt
import random
from scipy.optimize import curve_fit
import csv
from scipy.signal import find_peaks

###intensity ratio means transmission

def transmission_model(w, a, r):
    """
    ###refer to calculate transmission for parameters
    this is only the equation of the transmission, no input
    serves as a model for the curve fit
    """
    radius = 150e-6
    n_eff = 1.496
    L = (2 * np.pi) * radius  # Circumference in meters
    beta = (2 * np.pi * n_eff) / w # propogation constant (radians per meter)
    phi = beta * L  # Phase shift in radians
    numerator = a**2 - (2 * r * a * np.cos(phi)) + r**2
    denominator = 1 - (2 * a * r * np.cos(phi)) + (r * a)**2  
    return numerator / denominator

def calculate_transmission(a=0.85, r=0.85, n_eff=1.496, radius=150e-6, wavelength_range=(760e-9, 765e-9), num_points=10000):
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

def fwhm(a, r):
    radius = 150e-6
    L = (2 * np.pi) * radius  # Circumference in meters
    wave_res = 760.07911e-9
    n_g = 1.5
    numerator = (1- r * a) * wave_res **2
    denominator = (np.pi * n_g * L) * np.sqrt(r * a)
    return numerator /denominator
    




def plot_results(wavelengths, intensity_ratios, a, r):
    """
    Plots the original noisy data and the best fit curve.
    """
    fitted_intensities = transmission_model(wavelengths, a, r)
    
    plt.figure(figsize=(8, 5))
    plt.plot(wavelengths * 1e9, intensity_ratios, label='Noisy Data', color='b', alpha=0.5)
    plt.plot(wavelengths * 1e9, fitted_intensities, label= f'Best Fit: a={a}, r={r}', color='r', linewidth=2)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title('Transmission vs Wavelength with Best Fit')
    plt.legend()
    plt.grid()
    plt.show()


def find_minima(wavelenghts, intensity_ratios):
    """finds the minima wavelenghts"""
    opposite_transm = -np.array(intensity_ratios)
    peaks, _ = find_peaks(opposite_transm)

    minima_waves = wavelenghts[peaks]
    minima_transmissions = np.array(intensity_ratios)[peaks]
    return minima_waves, minima_transmissions

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

# Generate noisy transmission data
wavelengths, intensities = calculate_transmission()

# Save fake data to csv file
filename = 'rr_test.csv'
create_csv(filename, wavelengths, intensities)

# Fit the model to the data
a, r = fit_transmission(wavelengths, intensities)
print(f"Optimal values: a={a}, r={r}")

#Find FWHM
width = fwhm(a, r)
print(f"FWHM: {width}")

plot_results(wavelengths, intensities, a, r)


##FIND PEAK
minima_wavelenghts, minima_transmissions = find_minima(wavelengths, intensities)

#graph with minima plotted
plot_minima(minima_wavelenghts, minima_transmissions)