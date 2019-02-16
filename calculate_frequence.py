# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:20:49 2019

@author: yuxi
"""

from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import glob
from skimage.io import imread
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import os
from keras import backend as K
from sklearn.cluster import MiniBatchKMeans,KMeans
from collections import Counter
from sklearn.decomposition import PCA
import lda
import time
#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from function import newtxt,newimagedata,create_plots,plot_confusion_matrix,cnn_model,cnn_model1,cnn_model2,cnn_model_select_feature

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import os
from keras.utils.vis_utils import plot_model
from sklearn.decomposition import PCA

#%%
#divide the images into many patches
count = 0
all_divide_image = np.zeros([1,30,30,3])
read_files = glob.glob("./2012raw_data/*.txt")
for name in read_files:
    image_idex = name.split("_")[4]
    image_subidex = image_idex.split(".")[0]
    name = name.split("\\")[1]
    name = name.split(".")[0]
    path_image = str('./2012image/')+name+str('.jpg')
    single_img = imread(path_image)
    divide_image = np.zeros([3468,30,30,3])
    count_patch = 0
    for i in range(68):
        for j in range(51):
            divide_image[count_patch,:,:,:] = single_img[j*30:(j+1)*30,i*30:(i+1)*30]
            count_patch = count_patch + 1
    all_divide_image = np.vstack((all_divide_image,divide_image))
    count = count + 1
    print("---image %d finished" % (count))

#%%
all_divide_image = all_divide_image[1:]
#%% get the output of CNN
model = cnn_model()
model.load_weights('model_weight/2012images-areas-7.5-50epoch.h5')
cnn_output = K.function([model.layers[0].input], [model.layers[19].output])

#%%
#all_output = cnn_output([all_divide_image])[0]
all_output = np.zeros([1,512])
for i in range(120):
    once_image = all_divide_image[i*3468:(i+1)*3468]
    once_output = cnn_output([once_image])[0]
    all_output = np.vstack((all_output,once_output))
    print("---image %d finished" % (i))
#%%
all_output = all_output[1:]
#%%PCA
#all_output_reduced = PCA(n_components=128).fit_transform(all_output)
pca = PCA(n_components=28)
pca.fit(all_output)
#%%
all_output_reduced = pca.transform(all_output)





#%%
kmeans = MiniBatchKMeans(n_clusters=500,
        random_state=0,
        batch_size=50000,
        max_iter=50).fit(all_output_reduced)
#kmeans = KMeans(n_clusters = 5000,random_state=0).fit(all_output)
#%%
code_book = kmeans.cluster_centers_
d = kmeans.predict(all_output_reduced) 
#%%
all_frequence = np.zeros([120,500])
for j in range(120):
    label_image = d[j*3468:(j+1)*3468]
    k = label_image.tolist()
    h = Counter(k)
    frequence_image = np.zeros(500)
    for i in range(500):
        frequence_image[i] = h[i]
    all_frequence[j] = frequence_image
    print("---image %d finished" % (j))
    
#%%
all_frequence = all_frequence.astype(np.int32)
#%%
model = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
model.fit(all_frequence)
#%%
topic_word = model.topic_word_
#%%
f, ax= plt.subplots(5, 1, figsize=(10, 16), sharex=True)
for i, k in enumerate([0, 1, 2, 3,4]):
    ax[i].stem(topic_word[k,:], linefmt='b-',
               markerfmt='bo', basefmt='w-')
    ax[i].set_xlim(0,500)
    ax[i].set_ylim(0, 0.05)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Category {}".format(k))
    ax[i].set_xlabel("patch")



plt.tight_layout()
plt.show()
#%%
doc_topic = model.doc_topic_
#%%
f, ax= plt.subplots(6, 1, figsize=(5, 16), sharex=True)
for i, k in enumerate([0, 1, 2, 3, 4,5]):
    ax[i].stem(doc_topic[k,:], linefmt='r-',
               markerfmt='ro', basefmt='w-')
    ax[i].set_xlim(-1, 6)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel("Prob")
    ax[i].set_title("Image {}".format(k))
    ax[i].set_xlabel("category")



plt.tight_layout()
plt.show()

#%%
res = np.argmax(topic_word, axis=0)
label_patch = np.zeros(416160)
for i in range(416160):
    center_code = d[i]
    label_patch[i] = res[center_code]



#%% LDA model
def LDA_output(input_patch):
    #get the output of CNN
    cnn_feature = cnn_output([input_patch])[0]
    #PCA
    feature_reduced = pca.transform(cnn_feature)
    #find the center in the codebook
    code_center = int(kmeans.predict(feature_reduced))
    #find the category of input
    category = np.argmax(topic_word[:,code_center])
    return category
    
#%% result of image segmentation
label_patch = label_patch.astype(int)
testimage = Image.new('RGB', (2048,1536))
count_patch = 3468
for c in range(0,2048,30):
    for r in range(0,1536,30):
        
        if(label_patch[count_patch]==0):
            testimage.paste((0,0,255),[c,r,c+30,r+30])
        elif(label_patch[count_patch]==1):
            testimage.paste((105,105,105),[c,r,c+30,r+30])
        #elif(label_patch[count_patch]==2):
            #testimage.paste((169,169,169),[c,r,c+30,r+30])
        elif(label_patch[count_patch]==2):
            testimage.paste((255,0,0),[c,r,c+30,r+30])
        elif(label_patch[count_patch]==3):
            testimage.paste((0,255,0),[c,r,c+30,r+30])
        else:
            testimage.paste((255,255,0),[c,r,c+30,r+30])
        count_patch = count_patch + 1  

testimage.save('test.jpg')
#%%find the precent areas of coral
def percent_coral(testimage):
    count = 0
    #blue = np.array([0,0,255], dtype=np.uint8)
    test = np.uint8(testimage)
    pixel = np.zeros([1,1,3],dtype = np.uint8)
    for r in range(0,test.shape[0]):
        for c in range(0,test.shape[1]):
            pixel = test[r,c,:]
            #if(np.all(pixel == blue)):
            if((pixel[0]==0)and(pixel[1]==0)):
                count = count +1
    all_pixel = test.shape[0]*test.shape[1]
    coral_percent = (count/all_pixel)*100 
    coral_percent = round(coral_percent,2)
    return coral_percent       
        
#%%
windowsize_r = 30
windowsize_c = 30
#image_path = ("./2012image/201208172_T-12-46-15_Dive_01_017.jpg")
image_files  = glob.glob("./test_image/*")
count = 0
for name in image_files:
    
    name_piece = name.split("\\")[1]
    name_image = name_piece.split(".")[0]
    path_image = './test_image/'+str(name_image)+'.jpg'
    testimage = Image.new('RGB', (2048,1536))
    test_image = imread(path_image)
    start_time = time.time()
    for r in range(0,test_image.shape[0] - windowsize_r+1, 6):
        for c in range(0,test_image.shape[1] - windowsize_c+1, 6):
            window = test_image[r:r+windowsize_r,c:c+windowsize_c]
            window = window.reshape(-1,30,30,3)
            y_pred = LDA_output(window)
            if(y_pred==0):#CCA---red
                testimage.paste((255,0,0),[c,r,c+windowsize_c,r+windowsize_r])
            elif(y_pred==1):#coral---blue
                testimage.paste((169,169,169),[c,r,c+windowsize_c,r+windowsize_r])
            elif(y_pred==2):#ROC---deep gray
                testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
            elif(y_pred==3):#Ana---green
                testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
            #elif(y_pred==4):#DCP---slight gray
                #testimage.paste((105,105,105),[c,r,c+windowsize_c,r+windowsize_r])
            else:#others---yellow
                testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])
    count = count+1
    percent = percent_coral(testimage)
    testimage.save('./result/'+str(name_image)+' result'+str(percent)+'.jpg')
    print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
    print("there are {0}% coral in this image".format(percent))
#%%
first_list  = glob.glob("./all-image/*")
for file in first_list:
    file_image = file.split("\\")[1]
    raw_image_files = glob.glob('./all-image/'+str(file_image)+'/*.jpg')
    count = 0
    for name in raw_image_files:
        name_piece = name.split("\\")[1]
        name_image = name_piece.split(".")[0]
        path_image = './all-image/'+str(file_image)+'/'+str(name_image)+'.jpg'
        testimage = Image.open(path_image)
        test_image = imread(path_image)
        start_time = time.time()
        for r in range(0,test_image.shape[0] - windowsize_r+1, 6):
            for c in range(0,test_image.shape[1] - windowsize_c+1, 6):
                window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                window = window.reshape(-1,30,30,3)
                pred = model.predict(window,verbose=2)
                y_pred = np.argmax(pred,axis=1)
                if(y_pred==0):#coral---blue
                    testimage.paste((0,0,255),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==1):#DCP---slight gray
                    testimage.paste((105,105,105),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==2):#ROC---deep gray
                    testimage.paste((169,169,169),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==3):#CCA---red
                    testimage.paste((255,0,0),[c,r,c+windowsize_c,r+windowsize_r])
                elif(y_pred==4):#Ana---green
                    testimage.paste((0,255,0),[c,r,c+windowsize_c,r+windowsize_r])
                else:#others---yellow
                    testimage.paste((255,255,0),[c,r,c+windowsize_c,r+windowsize_r])
        count = count+1
        percent = percent_coral(testimage)
        testimage.save('./all-image/'+str(file_image)+'/'+str(name_image)+' result'+str(percent)+'.jpg')
        print(("---image%d finished in %s seconds ---" % (count,(time.time()-start_time))))
        print("there are {0}% coral in this image".format(percent))
    print("--------------------images in the {} finished -------------".format(file_image) )



















