import torch

import copy
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split

from torch import nn, optim

import torch.nn.functional as F
#from arff2pandas import a2p
from scipy.io import arff


train_data, meta = arff.loadarff(".\Data-20241013T093531Z-001\Data\ECG5000_unzipped\ECG5000_TRAIN.arff")
test_data, meta1 = arff.loadarff(".\Data-20241013T093531Z-001\Data\ECG5000_unzipped\ECG5000_TEST.arff")

train = pd.DataFrame(train_data)
test = pd.DataFrame(test_data)
df = pd.concat([train, test], ignore_index=True)
df = df.sample(frac=1.0)

new_columns = list(df.columns)
new_columns[-1] = 'target'
df.columns = new_columns
df['target'] = df['target'].astype(int)
# print(df.target.value_counts())


df.to_csv('.\Data-20241013T093531Z-001\EGC.csv')


# normal_df = df[df.target == CLASS_NORMAL].drop(labels='target', axis=1)
# anomaly_df = df[df.target != CLASS_NORMAL].drop(labels='target', axis=1)