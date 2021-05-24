import torch
import os
import numpy as np

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    elif delim == 'comma':
    	delim = ','
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [i for i in line]
            data.append(line)
    return np.asarray(data)


delim = 'comma'
path = os.path.join(os.getcwd(),'annotations.dat')
print(path)
data = read_file(path,delim)
print(data.shape)
cam1_data = [row for row in data if row[1] == 1]
print(cam1_data.shape)