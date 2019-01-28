# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:35:07 2019

@author: Xi Yu
"""

from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.io import imread
from function import shuffle
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import time
from keras import backend as K
from sklearn.cluster import MiniBatchKMeans


from skimage import color
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float
from skimage.measure import block_reduce
from function import newtxt,newimagedata,create_plots,plot_confusion_matrix,cnn_model,cnn_model1,cnn_model2
 #%%
img_path = './2012image/201208172_T-12-46-15_Dive_01_017.jpg'
img = imread(img_path)
#%%
divide_image = np.zeros([3468,30,30,3])
for i in range(51):
    for j in range(68):
        divide_image[(i+1)*j,:,:,:] = img[i*30:(i+1)*30,j*30:(j+1)*30]
        
#%%
#K.set_learning_phase(1) #set learning phase
model = cnn_model()
model.load_weights('model_weight/2012images-areas-7.5-50epoch.h5')
cnn_output = K.function([model.layers[0].input], [model.layers[19].output])
f1 = cnn_output([divide_image])[0]
#%%
kmeans = MiniBatchKMeans(n_clusters=300,
        random_state=0,
        batch_size=128,
        max_iter=10).fit(f1)
#%%
code_book = kmeans.cluster_centers_






