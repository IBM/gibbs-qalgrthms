import numpy as np
import os.path

#------- save the array into file ------- 
def savedata(M,filename):
    np.save(filename+'.npy', M)
    return 0

#------- load the array from the file ------- 
def loaddata(filename):
    M = np.load(filename+'.npy',allow_pickle=True)
    return M

#-------- check if the file exists ------- 
def exist(filename):
    return os.path.isfile(filename+'.npy')