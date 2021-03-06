# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 11:19:43 2018

@author: Xi Yu
"""
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
from skimage.io import imread  
from function import shuffle      
#%%
def generate_data(x1,y1,x2,y2): 
    if((x1!=x2) and (y1!=y2)):
        k = (y1-y2)/(x1-x2)
        b = y1-k*x1
        if(x1>x2):
            a1=x1
            b1=x2
            x2=a1
            x1=b1
        else:
            x1=x1
            x2=x2
        x = np.arange(x1+15,x2,15)
        size = x.shape[0]
        y = np.int32(k*x + b)
        
    elif((x1==x2) and (y1!=y2)):
        if(y1>y2):
            a1=y1
            a2=y2
            y2=a1
            y1=a2
        else:
            y1=y1
            y2=y2
        y = np.arange(y1+15,y2,15)
        size = y.shape[0]
        x = np.zeros(size)
        for i in range(size):
            x[i]=x1
            
    elif((x1!=x2) and (y1==y2)):
        if(x1>x2):
            a1=x1
            a2=x2
            x2=a1
            x1=a2
        else:
            x1=x1
            x2=x2
        x = np.arange(x1+15,x2,15)
        size = x.shape[0]
        y = np.zeros(size)
        for i in range(size):
            y[i]=y1
    
    return x,y,size
#%%
def generate_raw_data(path_txt,txt_name):
    #path_image = "./201208172_T-12-58-58_Dive_01_041.jpg"
    #path_txt = "./2012area/201208172_T-13-13-48_Dive_01_051.txt"
    #txt_name = '201208172_T-13-13-48_Dive_01_051'
    txtfile = open(path_txt)
    areaNum = int(txtfile.readlines()[2])
    all_point = areaNum*2
    txtfile.close()
    
    txtfile = open(path_txt)
    data = txtfile.readlines()[3:(areaNum+1)*3]
    corrdinate = np.zeros([areaNum*2,2],dtype = np.int)
    for i in range(areaNum):
        data_point = data[3*i+1:3*i+3]
        corrdinate[2*i,0] = data_point[0].split(',')[0]
        corrdinate[2*i,1] = data_point[0].split(',')[1]
        corrdinate[2*i+1,0] = data_point[1].split(',')[0]
        corrdinate[2*i+1,1] = data_point[1].split(',')[1]
    txtfile.close()
    
    label = np.zeros(areaNum*2,dtype = np.int)
    txtfile = open(path_txt)
    name_label = txtfile.readlines()[-(7+areaNum):-7]
    for j in range(areaNum):
        name_element = name_label[j].split(',')[5]
        na = name_element.split('"')[1]
        if ((na=="Agalg") or (na=="Aga") or (na=="Agaf") or (na=="Col") or (na=="Helc") or (na=="Lep") or (na=="Mdrc") or (na=="Mdrsa") or (na=="Mdrcb") or (na=="Mdrsd") or (na=="Mdrsf") or (na=="Mdrsp") or (na=="Man") or (na=="Mon") or (na=="Ocu") or (na=="Sco") or (na=="Sol") or (na=="Ste") or (na=="STY")):
            label[2*j] = 0
            label[2*j+1] = 0
        elif (na=="DCP"):
            label[2*j] = 1
            label[2*j+1] = 1
        elif (na=="ROC"):
            label[2*j] = 2
            label[2*j+1] = 2
        elif ((na=="CCA") or (na=="Amph") or (na=="Bot") or (na=="Haly") or (na=="Kal") or (na=="Mar") or (na=="PEY") or (na=="RHO") or (na=="RHbl")) :
            label[2*j] = 3
            label[2*j+1] = 3
        elif ((na=="Ana") or (na=="Cau") or (na=="Caup") or (na=="Caur") or (na=="Caus") or (na=="Chae") or (na=="CHL") or (na=="Cod") or (na=="Codin") or (na=="Hal") or (na=="Halc") or (na=="Hald") or (na=="Halt") or (na=="Micr") or (na=="Ulva") or (na=="Venv") or (na=="Verp")):
            label[2*j] = 4
            label[2*j+1] = 4 
        else:
            label[2*j] = 5
            label[2*j+1] = 5
            txtfile.close()
    txtfile.close()
    with open(str('2015_boundary-data/')+txt_name+str('.txt'), 'w+') as newtxtfile:
            newtxtfile.write(txt_name+str('\n'))
            newtxtfile.write('x,y,label\n')
            for i in range(all_point):
                x = corrdinate[i,0]
                y = corrdinate[i,1]
                z = label[i]
                newtxtfile.write('{0},{1},{2}\n'.format(x,y,z))
            newtxtfile.close()
#%%generate raw processing data
txt_name = glob.glob("./2015area/*.txt")
count = 1
for name in txt_name:
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    path_txt = str('./2015area/')+image_name+str('.txt')
    generate_raw_data(path_txt,image_name)
    print("Image {0} is finish ".format(count))
    count = count+1
#%%
def generate_all_data(new_txt_name):
    #new_txt_name = '201208172_T-12-55-49_Dive_01_035'
    new_path_txt = str('./2015-image-data - Copy/')+new_txt_name+ str('.txt')
    new_txt_file = open(new_path_txt)
    data_new = new_txt_file.readlines()[2:]
    size_data = len(data_new)
    pointNum = int(size_data/2)
    new_txt_file.close()
    for m in range(pointNum):
        dat = data_new[2*m].split(',')
        dat1 = data_new[2*m+1].split(',')
        x1 = int(float(dat[0]))
        y1 = int(float(dat[1]))
        x2 = int(float(dat1[0]))
        y2 = int(float(dat1[1]))
        label = int(dat[2])
        x,y,size = generate_data(x1,y1,x2,y2)
        for i in range(size):
            corrdinate_x = x[i]
            corrdinate_y = y[i]
            label_point = label
            writ = open(new_path_txt,"a")
            writ.write('{0},{1},{2}\n'.format(corrdinate_x,corrdinate_y,label_point))
            writ.close()
    
#%%
new_txt_name = glob.glob("./2015-image-data - Copy/*.txt")
all_data_count = 1
for name in new_txt_name:
    image_name = name.split("\\")[1]
    image_name = image_name.split(".")[0]
    #path_txt = str('./2012area/')+image_name+str('.txt')
    generate_all_data(image_name)
    print("Image {0} is finish ".format(all_data_count))
    all_data_count = all_data_count+1
#%%
def generate_signal_imagedata(path_txt,path_image,image_x,image_y):
    txt_file = open(path_txt)
    text = txt_file.readlines()[2:]
    count_points = len(text)
    
    crop_length = 30
    crop_width = 30
    all_image = np.zeros([count_points,crop_length,crop_width,3],dtype=np.uint8)
    label = np.zeros(count_points,dtype=np.int)
    crop_x = int(crop_length/2)
    crop_y = int(crop_width/2)
    image = imread(path_image)
    for i in range(count_points):
        text_piece = text[i]
        text_element = text_piece.split(',')
        l_x = int(float(text_element[0]))
        l_y = int(float(text_element[1]))
        label[i] = int(text_element[2])
        if(l_x-crop_x <0):
            l_x = crop_x
        if(l_y-crop_y <0):
            l_y = crop_y
        if(l_x+crop_x >image_y):
            l_x = image_y-15
        if(l_y+crop_y >image_x):
            l_y =image_x-15
        all_image[i,:,:,:] = image[l_y-15:l_y+15,l_x-15:l_x+15]
    txt_file.close()
    return all_image,label

#%%transfer the image into four dimentation data
def read_generate_data(txtfolder_name,imagefloder_name,image_x,image_y):
    all_count = 1
    all_label = np.zeros(1,dtype=np.int)
    image_data = np.zeros([1,30,30,3],dtype=np.uint8)
    read_files = glob.glob(str('./')+txtfolder_name+str('/*.txt'))
    for name in read_files:
        name = name.split("\\")[1]
        name = name.split(".")[0]
        path_txt = str('./')+txtfolder_name+str('/')+name+str('.txt')
        path_image = str('./')+ imagefloder_name+str('/')+name+str('.jpg')
        new_image_data,label = generate_signal_imagedata(path_txt,path_image,image_x,image_y)
        all_label = np.hstack((all_label,label))
        image_data = np.vstack((image_data,new_image_data))
        print("Image {0} is finish ".format(all_count))
        all_count = all_count+1
    final_data = image_data[1:,:,:,:]
    final_label = all_label[1:]
    return final_data,final_label
#%%read new data 2015 area data
imagefloder_name = '2015image'
new_txtfolder_name = '2015_boundary-data'
new_final_data,new_final_label = read_generate_data(new_txtfolder_name,imagefloder_name,2736,3648)
#%%read previous data 2015 image data
old_txtfolder_name = '2015-image-data'
old_final_data,old_final_label = read_generate_data(old_txtfolder_name,imagefloder_name,2736,3648)
#%%2012 image data
data_2012_imagefloder_name = '2012image'
data_2012_txtfolder_name = '2012data_label augmentation(m=300)'
data2012_data,data2012_label = read_generate_data(data_2012_txtfolder_name,data_2012_imagefloder_name,1536,2048)
#data2012_data = data2012_data[0:30000]
#data2012_label = data2012_label[0:30000]
#%%2012 area data
data2012area_name = '2012new - Copy'
data2012area_data,data2012area_label = read_generate_data(data2012area_name,data_2012_imagefloder_name,1536,2048)
#%%
all_data2012 = np.vstack((data2012_data,data2012area_data))
all_2012_label = np.hstack((data2012_label,data2012area_label))
#%%
new_data = new_final_data[0:1000]
new_label = new_final_label[0:1000]
x#%%
final_data = np.vstack((all_data2012,new_data))
#final_data = np.vstack((data2012_data,new_data))
#final_data = old_final_data
#final_label = np.hstack((old_final_label,data2012area_label))
final_label = np.hstack((all_2012_label,new_label))
#final_label = old_final_label
#%%
shuffle_data,shuffle_label = shuffle(final_data,final_label)
#%%\    data_shape = data.shape
data = all_data2012
label = all_2012_label
data_shape = data.shape
data = data.reshape(-1,data_shape[1]*data_shape[2]*data_shape[3])
#%%
Z = np.column_stack((data,label))
np.random.shuffle(Z)
#%%
fully_data = Z[:,:-1]
shuffle_label = Z[:,-1]
shuffle_data = fully_data.reshape(-1,30,30,3)

#%%
coral_count = 0
dcp_count = 0
roc_count = 0
red_count = 0
green_count = 0
other_count = 0
#label_count = len(final_label)
for i in final_label:
    if(i==0):
        coral_count = coral_count+1
    if(i==1):
        dcp_count = dcp_count+1
    if(i==2):
        roc_count = roc_count+1
    if(i==3):
        red_count = red_count+1
    if(i==4):
        green_count = green_count+1
    if(i==5):
        other_count = other_count+1

#%%











