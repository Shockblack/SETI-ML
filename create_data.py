#-----------------------------------------------------------------------
# File : create_data.py
#
# Programmer: 
#       github.com/zzheng18
#       Aiden Zelakiewicz
# 
# Creating synthetic h5 datafiles for waterfall plots with signals and
# RFI noise in half.
# 
# Credit:
#   Code base thanks to previous SETI intern's repository found at
#   https://github.com/zzheng18/SETIGAN
# 
#-----------------------------------------------------------------------

# Start with importing the necessary packages
import numpy as np
import setigen as stg
import h5py
from astropy import units as u
from tqdm import trange

# Determine the data filepath
data_path = '/datax/scratch/zelakiewicz/'
datafile_name = 'synthetic_30000_6_band.h5'

# Slopes of the drift rates
drift_rates = [0.3, 0.15, 0.1, -0.1, -0.15, -0.3]

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
        for i in trange(2500):

            # Randomizing signal width
            random_width = np.random.uniform(50, 60)

            # Changing random start based on neg or pos drift rate
            if dr > 0:
                random_start = np.random.uniform(-10, 70)
            else:      
                random_start = np.random.uniform(60, 140)

            # Creating the on target frame.
            on_frame = stg.Frame(fchans=128*u.pixel, # number of frequency samples
                          tchans=128*u.pixel, # number of time samples
                          df=2.7939677238464355*u.Hz, # frequency resolution
                          dt=18.253611008*u.s, # time resolution
                          fch1=6095.214842353016*u.MHz) # min/max frequency

            # Adding noise to the on frame
            noise = on_frame.add_noise(x_mean=10, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            # Adding a signal to the on frame
            signal = on_frame.add_signal(stg.constant_path(f_start=on_frame.get_frequency(index=random_start), \
                                    drift_rate=dr*u.Hz/u.s), \
                                    stg.constant_t_profile(level=5), \
                                    stg.gaussian_f_profile(width=random_width*u.Hz), \
                                    stg.constant_bp_profile(level=1))
            
            # Creating first off target frame
            off_frame_1 = stg.Frame(fchans=128*u.pixel, tchans=21*u.pixel, df=2.7939677238464355*u.Hz, \
                            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)
                            
            noise = off_frame_1.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            # Second off target frame
            off_frame_2 = stg.Frame(fchans=128*u.pixel, tchans=21*u.pixel, df=2.7939677238464355*u.Hz, \
                          dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

            noise = off_frame_2.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            off_frame_3 = stg.Frame(fchans=128*u.pixel, tchans=23*u.pixel, df=2.7939677238464355*u.Hz, \
                          dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz) 
            
            noise = off_frame_3.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

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
            # data_list.append(img)
            # label_list.append(dr)

        # Looping over number of figures with RFI for given drift rate
        for i in trange(2500):

            # Randomizing signal width
            random_width = np.random.uniform(50, 60)

            # Changing random start based on neg or pos drift rate
            if dr > 0:
                random_start = np.random.uniform(-10, 70)
            else:      
                random_start = np.random.uniform(60, 140)

            # Creating the on target frame.
            on_frame = stg.Frame(fchans=128*u.pixel, # number of frequency samples
                          tchans=128*u.pixel, # number of time samples
                          df=2.7939677238464355*u.Hz, # frequency resolution
                          dt=18.253611008*u.s, # time resolution
                          fch1=6095.214842353016*u.MHz) # min/max frequency

            # Adding noise to the on frame
            noise = on_frame.add_noise(x_mean=10, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            # Adding a signal to the on frame
            signal = on_frame.add_signal(stg.constant_path(f_start=on_frame.get_frequency(index=random_start), \
                                    drift_rate=dr*u.Hz/u.s), \
                                    stg.constant_t_profile(level=5), \
                                    stg.gaussian_f_profile(width=random_width*u.Hz), \
                                    stg.constant_bp_profile(level=1))
            
            # Creating first off target frame
            off_frame_1 = stg.Frame(fchans=128*u.pixel, tchans=21*u.pixel, df=2.7939677238464355*u.Hz, \
                            dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)
                            
            signal = off_frame_1.add_signal(stg.simple_rfi_path(f_start=on_frame.get_frequency(index=np.random.uniform(0, 128)),
                                     drift_rate=0*u.Hz/u.s,
                                     spread=np.random.uniform(0, 30)*u.Hz,
                                     spread_type='uniform',
                                     rfi_type='random_walk'),
                                     stg.constant_t_profile(level=1),
                                     stg.box_f_profile(width=20*u.Hz),
                                     stg.constant_bp_profile(level=1))
            
            noise = off_frame_1.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            # Second off target frame
            off_frame_2 = stg.Frame(fchans=128*u.pixel, tchans=21*u.pixel, df=2.7939677238464355*u.Hz, \
                          dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)
            
            signal = off_frame_2.add_signal(stg.simple_rfi_path(f_start=on_frame.get_frequency(index=np.random.uniform(0, 128)),
                                     drift_rate=0*u.Hz/u.s,
                                     spread=np.random.uniform(0, 30)*u.Hz,
                                     spread_type='uniform',
                                     rfi_type='random_walk'),
                                     stg.constant_t_profile(level=1),
                                     stg.box_f_profile(width=20*u.Hz),
                                     stg.constant_bp_profile(level=1))

            noise = off_frame_2.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

            off_frame_3 = stg.Frame(fchans=128*u.pixel, tchans=23*u.pixel, df=2.7939677238464355*u.Hz, \
                          dt=18.253611008*u.s, fch1=6095.214842353016*u.MHz)

            signal = off_frame_3.add_signal(stg.simple_rfi_path(f_start=on_frame.get_frequency(index=np.random.uniform(0, 128)),
                                     drift_rate=0*u.Hz/u.s,
                                     spread=np.random.uniform(0, 30)*u.Hz,
                                     spread_type='uniform',
                                     rfi_type='random_walk'),
                                     stg.constant_t_profile(level=1),
                                     stg.box_f_profile(width=20*u.Hz),
                                     stg.constant_bp_profile(level=1))
            
            noise = off_frame_3.add_noise(x_mean=12, x_std = 1, noise_type = np.random.choice(['normal', 'chi2']))

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

    