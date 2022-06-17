#----------------------------------------------------------
# File : custom_dataloader.py
#        Aiden Zelakiewicz
# 
# A custom dataloader class for Pytorch NN model creation.
# 
# 
#----------------------------------------------------------

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py