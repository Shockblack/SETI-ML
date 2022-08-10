#-----------------------------------------------------------------------
# File : create_data.py
#
# Programmer: 
#       Aiden Zelakiewicz
#       Zihe Zhang
# 
# Creating synthetic h5 datafiles for waterfall plots with signals and
# RFI noise in half. Signals are linear and constant with randomized
# intensity. Uses Bryan Brzycki's package for synthetic signal creation
# SETIGEN (https://github.com/bbrzycki/setigen).
# 
# Credit:
#   Code base thanks to previous SETI intern Zihe Zhang's notebook found
#   at https://github.com/zzheng18/SETIGAN/blob/master/Generate_data.ipynb. 
#   This code is an implementation of one code block from her notebook.
# 
#-----------------------------------------------------------------------

# Start with importing the necessary packages
import numpy as np
import setigen as stg
import h5py
from astropy import units as u
from tqdm import trange

# Total amount of data to generate
total_data = 15000
data_per_iter = total_data/12 # Number of data points per iteration | 6 slopes for both noisy and clean = 12

# Determine the data filepath
data_path = '/datax/scratch/zelakiewicz/15k'
datafile_name = 'synthetic_'+str(total_data)+'_6_band.h5'


# Slopes of the drift rates
drift_rates = [0.05, 0.15, 0.1, -0.1, -0.15, -0.05]

x_mean_on = 5

x_mean_off_type = 'random'

if x_mean_off_type == 'constant':
    x_mean_off = 12

def create_off_frame(x_mean, x_std, tchans, rfi):
    """Creates an off frame of synthetic waterfall data. Function is meant
    to clean up code loop and make the script more readable.

    Parameters
    ----------
    x_mean : int, float
        The mean of the noise distribution.
    x_std : int, float
        The standard deviation of the noise distribution.
    tchans : float
        The number of time channels in the data, equates to number of pixels.
    rfi : bool
        If True, adds RFI noise to the waterfall plot.

    Returns
    -------
    stg.Frame
        The SETIGEN output frame.
    """

    # Create the off frame
    frame = stg.Frame(fchans=128*u.pixel, tchans=tchans*u.pixel, df=2.7939677238464355*u.Hz, \
                            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

    # Add background noise to the frame
    noise = frame.add_noise(x_mean=x_mean, x_std = x_std, noise_type = np.random.choice(['normal', 'chi2']))

    # If rfi is True, add RFI noise to the frame
    if rfi:
        signal = frame.add_signal(stg.simple_rfi_path(f_start=on_frame.get_frequency(index=np.random.uniform(0, 128)), \
                            drift_rate=0*u.Hz/u.s, spread=np.random.uniform(0, 30)*u.Hz, spread_type='uniform', \
                            rfi_type='random_walk'), stg.constant_t_profile(level=1), \
                            stg.box_f_profile(width=20*u.Hz), stg.constant_bp_profile(level=1))

    return frame

def create_on_frame(x_mean, x_std, snr, signal_start, drift_rate, signal_width):
    """Creates an on frame of synthetic waterfall data. Function is meant
    to clean up code loop and make the script more readable. Adds a constant
    drifting signal to the frame with no rfi in it's noise background.

    Parameters
    ----------
    x_mean : int, float
        The mean of the noise distribution.
    x_std : int, float
        The standard deviation of the noise distribution.
    snr : int, float
        The signal to noise ratio of the signal.
    signal_start : int
        The starting location of the signal in pixels.
    drift_rate : int, float
        The drift rate of the signal.
    signal_width : int, float
        The width of the signal.

    Returns
    -------
    stg.Frame
        The SETIGEN output frame.
    """


    # Create the on frame, 128 by 128 pixels
    frame = stg.Frame(fchans=128*u.pixel, tchans=128*u.pixel, df=2.7939677238464355*u.Hz, \
                            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

    # Add background noise to the frame
    noise = frame.add_noise(x_mean=x_mean, x_std = x_std, noise_type = np.random.choice(['normal', 'chi2']))

    # Add signal to the frame
    signal = frame.add_constant_signal(f_start=on_frame.get_frequency(index=signal_start), \
                drift_rate=drift_rate*u.Hz/u.s, level = on_frame.get_intensity(snr=snr), width = signal_width*u.Hz)

    return frame

# Start creating the synthetic data files
with h5py.File(data_path + datafile_name, 'w') as hf:

    data_list = []
    label_list = []

    # Counter variable
    dr_num = 0
    # Loop over the different drift rates
    for k in range(len(drift_rates)):

        dr = drift_rates[k]
        dr_num += 1
        label_ind = k

        # Looping over number of noiseless figures for given drift rate
        for i in trange(data_per_iter):

            if x_mean_off_type == 'random':
                x_mean_off = np.random.randint(7, 9)

            # Randomizing signal width
            random_width = np.random.uniform(20, 30)

            # Changing random start based on neg or pos drift rate
            if dr > 0:
                random_start = np.random.randint(-10, 30)
            else:      
                random_start = np.random.randint(100, 140)

            # Creating the on target frame.
            on_frame = create_on_frame(x_mean_on, 1, 300, random_start, dr, random_width)
            
            # Creating first off target frame
            off_frame_1 = create_off_frame(x_mean_off, 1, 21, False)
            
            # Second off target frame
            off_frame_2 = create_off_frame(x_mean_off, 1, 21, False)

            # Third off target frame
            off_frame_3 = create_off_frame(x_mean_off, 1, 23, False)

            # group and data naming
            group_name = 'clean_'+str(i+1)+'_'+str(dr_num)
            grp = hf.create_group(group_name)

            on_frame._update_waterfall()
            img = on_frame.waterfall.data
            img[63:84,:] = off_frame_2.get_data().reshape(21,1,128)
            img[21:42,:] = off_frame_1.get_data().reshape(21,1,128)
            img[105:,:] = off_frame_3.get_data().reshape(23,1,128)

            grp.create_dataset('data', data=np.array(img, dtype=np.float32))
            grp.create_dataset('label', data=np.array(label_ind, dtype=np.int32))

        # Looping over number of figures with RFI for given drift rate
        for i in trange(data_per_iter):

            if x_mean_off_type == 'random':
                x_mean_off = np.random.randint(7, 9)

            # Randomizing signal width
            random_width = np.random.uniform(20, 30)

            # Changing random start based on neg or pos drift rate
            if dr > 0:
                random_start = np.random.randint(-10, 30)
            else:      
                random_start = np.random.randint(100, 140)

            # Creating the on target frame.
           # Creating the on target frame.
            on_frame = create_on_frame(x_mean_on, 1, 300, random_start, dr, random_width)
            
            # Creating first off target frame
            off_frame_1 = create_off_frame(x_mean_off, 1, 21, True)
            
            # Second off target frame
            off_frame_2 = create_off_frame(x_mean_off, 1, 21, True)

            # Third off target frame
            off_frame_3 = create_off_frame(x_mean_off, 1, 23, True)

             # group and data naming
            group_name = 'noisy_'+str(i+1)+'_'+str(dr_num)
            grp = hf.create_group(group_name)

            on_frame._update_waterfall()
            img = on_frame.waterfall.data
            img[63:84,:] = off_frame_2.get_data().reshape(21,1,128)
            img[21:42,:] = off_frame_1.get_data().reshape(21,1,128)
            img[105:,:] = off_frame_3.get_data().reshape(23,1,128)

            grp.create_dataset('data', data=np.array(img, dtype=np.float32))
            grp.create_dataset('label', data=np.array(label_ind, dtype=np.int32))
    
    hf.close()

    