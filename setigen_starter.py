#----------------------------------------------------------
# File : setigen_starter.py
#        Aiden Zelakiewicz
# 
# Python file to get used to SETIGEN and creating waterfall
# plots in python. 
# 
#----------------------------------------------------------

# Start by importing necessary packages
import matplotlib
import setigen as stg
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import time
import ipdb

matplotlib.use( 'tkagg' )

cmap_array = np.loadtxt('dusk_cm.txt')
cmap = matplotlib.colors.ListedColormap(cmap_array[::-1]/255.0)

# Creating a frame
frame = stg.Frame(fchans=128*u.pixel, tchans=128*u.pixel, df=2.7939677238464355*u.Hz, \
            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

# Adding background noise
noise = frame.add_noise(x_mean=10, noise_type='chi2')

# Injecting signal
signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=1), drift_rate=0.15*u.Hz/u.s), \
                            stg.constant_t_profile(level=frame.get_intensity(snr=40)), \
                            stg.gaussian_f_profile(width=10*u.Hz), \
                            stg.constant_bp_profile(level=1))

fig = plt.figure(figsize=(10, 6))

frame2 = stg.Frame(fchans=128*u.pixel, tchans=128*u.pixel, df=2.7939677238464355*u.Hz, \
            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)
            

noise2 = frame2.add_noise(x_mean=10, noise_type='chi2')


metadata = np.concatenate((frame.get_data()[:21],frame2.get_data()[21:42],\
                            frame.get_data()[42:63],frame2.get_data()[63:84],\
                            frame.get_data()[84:105],frame2.get_data()[105:]))

plt.imshow(metadata, aspect='auto', interpolation='none', cmap=cmap)

plt.colorbar()
plt.xlabel('Frequency (px)')
plt.ylabel('Time (px)')

plt.savefig('/Users/shockblack/Documents/Research/SETI/SETI-ML/samplePlot.png', bbox_inches='tight')
plt.show()