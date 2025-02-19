from matplotlib import pyplot as plt
import numpy as np
import math
import random
from scipy import optimize as opt

#equation 2

#### a and r BOTH AFFECT AMPLITUDE

a = 0.85 #single pass amplitude transmission (example value)
r = 0.85 # self coupling coefficient (example value)
n_eff = 1.496 #refractive index


#### RADIUS AFFECTS PERIOD OF WAVE
radius = 150e-6 #meters
L = (2 * np.pi) * radius #meters


wavelengths = np.linspace(760e-9, 765e-9, 10000) #50 different wavelenghts between 760nm-750nm


intensity_ratios = []

for w in wavelengths: #loop through differen values of w, calculates intensity for each
    beta = (2 * np.pi * n_eff) / w #propogation constant (radians per meter)
    phi = beta * L #radians
    numerator = a**2 - (2 * r * a * np.cos(phi)) + r**2
    denominator = 1 - (2 * a * r * np.cos(phi)) + (r*a)**2  
    intensity = numerator / denominator 

    #generate random noise 
    rand_num = random.uniform(0, intensity * 0.05) 
    if random.choice([True, False]):
        intensity += rand_num
    else:
        intensity -= rand_num
        
    intensity_ratios.append(intensity)



#plot 
plt.plot(wavelengths * 1e9, intensity_ratios)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.grid(True)
plt.show()



##need optimal values for a and k

