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
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import time

# Creating a frame
frame = stg.Frame(fchans=1024*u.pixel, tchans=32*u.pixel, df=2.7939677238464355*u.Hz, \
            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

noise = frame.add_noise(x_mean=10, noise_type='chi2')

signal = frame.add_signal(stg.constant_path(f_start=frame.get_frequency(index=200), drift_rate=2*u.Hz/u.s), \
                            stg.constant_t_profile(level=frame.get_intensity(snr=30)), \
                            stg.gaussian_f_profile(width=40*u.Hz), \
                            stg.constant_bp_profile(level=1))

fig = plt.figure(figsize=(10, 6))
frame.plot()
plt.savefig('samplePlot.png', bbox_inches='tight')

help(stg.Frame)