# -*- coding: utf-8 -*-
"""
Created on Tue May 22 17:24:10 2018

@author: yuxi
"""


# import the necessary packages
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.io import imread
from function import shuffle
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import time


from skimage import color
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries,find_boundaries
from skimage.util import img_as_float
from skimage.measure import block_reduce
#%%
#path_image = "./2012image/201208172_T-12-50-10_Dive_01_025.jpg"
path_image = "./2015image/201509022-T-11-23-45_032.jpg"

img = imread(path_image)
img = img_as_float(img)
#%%image segmentation with k-means(step1)
segments_slic = slic(img, n_segments=300, compactness=10, sigma=10)
print('SLIC number of segments: {}'.format(len(np.unique(segments_slic))))
fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()
#%%
def mark_point(x,y,label):
    if (label == 0):#Coral
        plt.plot(x,y,'bs')
    if (label == 1):#DCP
        plt.plot(x,y,'ks')
    if (label == 2):#ROC
        plt.plot(x,y,'kx')
    if (label == 3):#CCA
        plt.plot(x,y,'rs')
    if (label == 4):#Ana
        plt.plot(x,y,'gs')
    if (label == 5):#Others
        plt.plot(x,y,'ms')
    plt.axis('off')
    

def mark_generate_point(x,y,label):
    if (label == 0):#Coral
        plt.plot(x,y,'b^')
    if (label == 1):#DCP
        plt.plot(x,y,'k^')
    if (label == 2):#ROC
        plt.plot(x,y,'k^')
    if (label == 3):#CCA
        plt.plot(x,y,'r^')
    if (label == 4):#Ana
        plt.plot(x,y,'g^')
    if (label == 5):#Others
        plt.plot(x,y,'m^')
    plt.axis('off')


#%%generate samples: generate 10 points for each original points
def generate_sample(segments_slic,original_label,label,size,write_path):
    a = np.where(segments_slic==label)
    a_y = a[0]
    a_x = a[1]
    num_dime = a_x.shape[0]
    #size represent the number of the generate data
    f = np.random.randint(num_dime, size=size)
    corrdinate_x = np.zeros(size)
    corrdinate_y = np.zeros(size)
    for i,j in enumerate (f):
        corrdinate_x[i] = a_x[j]
        corrdinate_y[i] = a_y[j]
        writ = open(write_path,"a") #writ is a file object for write
        x = int(corrdinate_x[i])
        y = int(corrdinate_y[i])
        label = original_label
        writ.write('{0},{1},{2}\n'.format(x,y,label))
        writ.close()
#%%generate labels in one image
def generate_signal_imagedata(path_txt,path_image): 
    txt_file = open(path_txt)
    text = txt_file.readlines()[2:]
    count_points = 550
    crop_length = 30
    crop_width = 30
    all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
    label = np.zeros(count_points)
    crop_x = int(crop_length/2)
    crop_y = int(crop_width/2)
    image = imread(path_image)
    for i in range(count_points):
        text_piece = text[i]
        text_element = text_piece.split(',')
        l_x = int(text_element[0])
        l_y = int(text_element[1])
        label[i] = int(text_element[2])
        if(l_x-crop_x <0):
            l_x = crop_x
        if(l_y-crop_y <0):
            l_y = crop_y
        if(l_x+crop_x >3648):
            l_x = 3648-15
        if(l_y+crop_y >2736):
            l_y =2736-15
        all_image[i,:,:,:] = image[l_y-15:l_y+15,l_x-15:l_x+15]
    txt_file.close()
    return all_image,label
#%%generate labels in all images
count_points = 50
raw_txt_files = glob.glob("./2015-image-data/*.txt")
count = 0
for txt_name in raw_txt_files:
    start = time.time()
    name = txt_name.split("\\")[1]
    name = name.split(".")[0]
    path_image = str('./2015image/')+name+str('.jpg')
    txt_file = open(txt_name)
    text = txt_file.readlines()[2:]
    img = imread(path_image)
    img = img_as_float(img)
    #quantize the image with k-mean
    segments_slic = slic(img, n_segments=300, compactness=8, sigma=10)
    for i in range(count_points): 
       line_piece = text[i]
       list_element = line_piece.split(',')
       l_x = int(list_element[0])
       l_y = int(list_element[1])
       original_label = int(list_element[2])
       label = segments_slic[l_y,l_x]
       generate_sample(segments_slic,original_label,label,10,txt_name)
    txt_file.close()
    end = time.time()
    count = count+1
    print(("---image%d finished in %s seconds ---" % (count,(end - start))))
#%%
count_points = 50
count = 0
txt_path = './2015-image-data/201509022-T-11-16-17_001.txt'
img_path = './2015image/201509022-T-11-16-17_001.jpg'
img = imread(img_path)
img = img_as_float(img)
#quantize the image with k-mean
txt_file = open(txt_path)
text = txt_file.readlines()[2:]
segments_slic = slic(img, n_segments=300, compactness=10, sigma=1)
for i in range(count_points):
    line_piece = text[i]
    list_element = line_piece.split(',')
    l_x = int(list_element[0])
    l_y = int(list_element[1])
    original_label = int(list_element[2])
    label = segments_slic[l_y,l_x]
    generate_sample(segments_slic,original_label,label,10,txt_path)
txt_file.close()













#%%transfer the image into four dimentation data
all_label = np.zeros(1)
image_data = np.zeros([1,30,30,3],dtype=np.uint8)
read_files = glob.glob("./2012image/*.txt")
for name in read_files:
    name = name.split("\\")[1]
    name = name.split(".")[0]
    path_txt = str('./2012data_label augmentation(m=300)/')+name+str('.txt')
    path_image = str('./2012image/')+name+str('.jpg')
    new_image_data,label = generate_signal_imagedata(path_txt,path_image)
    all_label = np.hstack((all_label,label))
    image_data = np.vstack((image_data,new_image_data))
final_data = image_data[1:,:,:,:]
final_label = all_label[1:]
#final_data,final_label = shuffle(final_data,final_label)
#%%
def drew_label_points(image_name,n_se,comp,sig):
    path_image = str('./2015image/')+image_name+str('.jpg')
    path_txt = str('./2015test/')+image_name+str('.txt')
    #path_image = './2015image/201509022-T-11-16-17_001.jpg'
    #path_txt = './2015-image-data/201509022-T-11-16-17_001.txt'
    count_points = 50
    
    img = imread(path_image)
    img = img_as_float(img)
    #quantize the image with k-mean
    txt_file = open(path_txt)
    text = txt_file.readlines()[2:]
    segments_slic = slic(img, n_segments=n_se, compactness=comp, sigma=sig)
    for i in range(count_points):
        line_piece = text[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        label = segments_slic[l_y,l_x]
        generate_sample(segments_slic,original_label,label,10,path_txt)
    txt_file.close()
    
    geneate_count_points = 500
    txtfile = open(path_txt)
    all_lines = txtfile.readlines()[2:]
    lines = all_lines[0:50]
    generate_lines = all_lines[50:]
    fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
    ax.imshow(mark_boundaries(img, segments_slic,color=(1,1,0)))
    for i in range(count_points):
        line_piece = lines[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        mark_point(l_x,l_y,original_label)
    plt.savefig(str('./test1/')+ image_name+ str('_original_points.png'),bbox_inches='tight')
    
    for j in range(geneate_count_points):
        line_piece = generate_lines[j]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        mark_generate_point(l_x,l_y,original_label)   
    plt.savefig(str('./test1/')+ image_name+ str('_generate_points.png'),bbox_inches='tight')
    plt.close(fig)
#%%show the images with labeld pixels 
image_count = 120
count = 1
txt_name = glob.glob("./2015image/*.jpg")
for name in txt_name:
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    print("Image {0} is processing ".format(count))
    start_time = time.time()
    drew_label_points(image_name,300,8,10)
    #print("Image {0} finished ".format(count))
    print(("Image %d finished in %f seconds ---" % (count,(time.time()-start_time))))
    count = count+1
#%%
img_name = '201509022-T-11-23-45_032'
drew_label_points(img_name,300,10,10)

#%%
count = 0
name = "./2012new\\201208172_T-13-22-18_Dive_01_070.txt"
image_name = name.split("\\")[1]
image_name = image_name.split(".")[0]
print("Image {0} is processing ".format(count))
start_time = time.time()
drew_label_points(image_name)
print(("Image %d finished in %f seconds ---" % (count,(time.time()-start_time))))
#%%
hsv_data = np.zeros((66000,30,30,3))
for i in range(66000):
    hsv_data[i]= color.rgb2lab(final_data[i])

#%%
def label_data(image_name):
#image_name = '201509022-T-11-19-41_013'
    path_image = str('./2015image/')+image_name+str('.jpg')
    path_txt = str('./2015_boundary-data/')+image_name+str('.txt')
    #path_image = '201208172_T-12-58-58_Dive_01_041.jpg'
    #path_txt = '201208172_T-12-58-58_Dive_01_041.txt'
    #image_name = '201705213_T-15-21-21_102'
    img = imread(path_image)
    txtfile = open(path_txt)
    lines = txtfile.readlines()[2:]
    fig, ax = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
    ax.imshow(img)
    count = len(lines)
    for i in range(count):
        line_piece = lines[i]
        list_element = line_piece.split(',')
        l_x = int(list_element[0])
        l_y = int(list_element[1])
        original_label = int(list_element[2])
        mark_point(l_x,l_y,original_label)
    plt.savefig('./2015/'+image_name+'area_points.png',bbox_inches='tight')
    plt.close(fig)

#%%
area_name = glob.glob("./2015_boundary-data/*.txt")
for name in area_name:
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    label_data(image_name)














