#----------------------------------------------------------
# File : setigen_multiplot.py
#        Aiden Zelakiewicz
# 
# Python file to get used to SETIGEN and creating waterfall
# plots in python. 
# 
#----------------------------------------------------------

# Start by importing
from code import interact
from tracemalloc import start
import matplotlib
import setigen as stg
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import h5py
import ipdb

# Grabbing my custom colormap
cmap_array = np.loadtxt('dusk_cm.txt')
cmap = matplotlib.colors.ListedColormap(cmap_array[::-1]/255.0)

# For plotting
matplotlib.use( 'tkagg' )


# Create a function to produce synthetic waterfalls
def create_synth_waterfall(drift_rate = 0.15, start_pix = 1, ran_snr = False, default_snr = 40, f_pix = 128, t_pix = 128):

    # Check if snr is supposed to be random
    if ran_snr == True:
        snr = np.random.randint(20,60)
    else:
        snr = default_snr

    # Creating a splicing value
    sval = t_pix // 6

    # Start by creating the synthetic signal

    # The frame
    on_frame = stg.Frame(fchans=f_pix*u.pixel, tchans=t_pix*u.pixel, df=2.7939677238464355*u.Hz, \
            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

    # Adding background noise
    on_frame.add_noise(x_mean=10, noise_type='chi2')

    # Adding the synthetic signal
    on_frame.add_signal(stg.constant_path(f_start=on_frame.get_frequency(index=start_pix), drift_rate=drift_rate*u.Hz/u.s), \
                            stg.constant_t_profile(level=on_frame.get_intensity(snr=snr)), \
                            stg.gaussian_f_profile(width=10*u.Hz), \
                            stg.constant_bp_profile(level=1))

    # Off Signal
    off_frame = stg.Frame(fchans=f_pix*u.pixel, tchans=t_pix*u.pixel, df=2.7939677238464355*u.Hz, \
            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

    # Adding noise
    off_frame.add_noise(x_mean=10, noise_type='chi2')

    # Creating a data framework to return
    signal_data = np.concatenate((on_frame.get_data()[:sval],off_frame.get_data()[sval:2*sval],\
                            on_frame.get_data()[2*sval:3*sval],off_frame.get_data()[3*sval:4*sval],\
                            on_frame.get_data()[4*sval:5*sval],off_frame.get_data()[5*sval:]))

    # Returning signal_data
    return signal_data

# Create list of variables to iterate over
slopes = [0.3, 0.15, 0.1, -0.1, -0.15, -0.3]
intercepts = [0, 0, 0, 128, 128, 128]

# Empty list
plot_data = []

# Looping over columns
for i in range(6):

    # Determining the slope and intercept
    slope = slopes[i]
    intercept = intercepts[i]

    # Empty list for column data
    column_data = []
    
    # Looping over rows, applying a random intensity
    for k in range(6):
        # Plotting and appending
        matrix_data = create_synth_waterfall(drift_rate=slope, start_pix=intercept, ran_snr=True)
        plot_data.append(matrix_data)

    #plot_data.append(column_data)

fig = plt.figure(figsize=(12,12))

ax = [fig.add_subplot(6,6,i+1) for i in range(36)]

for axes in ax:
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_aspect('equal')

for k in range(36):

    plt.sca(ax[k])
    plt.imshow(plot_data[k], aspect='auto', interpolation='none', cmap=cmap)

fig.subplots_adjust(wspace=0, hspace=0)
#plt.colorbar()
plt.show()

print(len(plot_data))
print(len(plot_data[0]))
print(len(plot_data[4]))