# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:02:33 2019

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
from sklearn.decomposition import LatentDirichletAllocation
import lda
import time
#%%
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
K.set_learning_phase(1)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 
import os
from keras.utils.vis_utils import plot_model
from sklearn.decomposition import PCA
from function import cnn_model, generate_signal_imagedata, read_generate_data
#%%
## divide all images into many patches, 
## the size of the patch is 30*30, the size of image is 2048*1536
## so the size of all_divide_image array is (120*3468)*30*30*3
x = np.arange(0,1536 - 30+1, 30)
y = np.arange(0,2048 - 30+1, 30)
X,Y = np.meshgrid(x, y)
def fun(x,y):
    m = single_img[x:x+30,y:y+30]
    m = m.reshape(2700)
    return m

count = 0
all_divide_image = np.zeros([1,30,30,3])
read_files = glob.glob("./2012raw_data/*.txt")
for name in read_files:
    name = name.split("\\")[1]
    name = name.split(".")[0]
    path_image = str('./2012image/')+name+str('.jpg')
    single_img = imread(path_image)
    check_palindrome = np.frompyfunc(fun, 2, 1)
    zs= check_palindrome(np.ravel(X.T), np.ravel(Y.T))
    fs = np.concatenate(zs,axis=0).astype(np.uint16)
    divide_image = fs.reshape(-1,30,30,3)
    all_divide_image = np.vstack((all_divide_image,divide_image))
    count = count + 1
    print("---image %d finished" % (count))
#get the size of all_patch_image
all_patch_image = all_divide_image[1:]/255
print("finished construct patches")
#%%
#load the pre-train CNN model and regard the output of last convolutional layers as feature
model = cnn_model()
model.load_weights('model_weight/2012_raw.h5')
cnn_layer_model = Model(inputs=model.input,
                                     outputs=model.layers[19].output)

# get the feature of pre-train CNN model
all_cnn_output = cnn_layer_model.predict(all_patch_image,verbose=1)
print("get the result of CNN")
#%%
pca = PCA(n_components=28)
pca.fit(all_cnn_output)
#%%
#using PCA to reduce the dimension: from 512 to 28
all_output_reduced = pca.transform(all_cnn_output)
#%%
#using k-mean to reduce the size of codebook from 416160 to 500
kmeans = MiniBatchKMeans(n_clusters=500,
        random_state=0,
        batch_size=50000,
        max_iter=50).fit(all_output_reduced)

#%%
code_book = kmeans.cluster_centers_
d = kmeans.predict(all_output_reduced)
#calculate the frequence of each centers in the codebook matrix
all_frequence = np.zeros([120,256])
for j in range(120):
    label_image = d[j*3468:(j+1)*3468]
    k = label_image.tolist()
    h = Counter(k)
    frequence_image = np.zeros(256)
    for i in range(256):
        frequence_image[i] = h[i]
    all_frequence[j] = frequence_image
    print("---image %d finished" % (j))
#%%
all_frequence = all_frequence.astype(np.int32)
#model = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
LDA_model = LatentDirichletAllocation(n_components=5,max_iter=1000,random_state=0)
LDA_model.fit(all_frequence)
#%%
tt = LDA_model.fit_transform(all_frequence)
#%%
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components =2, verbose =1, random_state =0, angle =.99, init='pca')
tsne_lda = tsne_model .fit_transform(tt)
#%%
colormap = np .array(["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"
                      ])
#"#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
#                      "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
#                      "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"

lda_keys = []
for i in range(tt .shape[0]):
    lda_keys += tt[i] .argmax(),

num_example = len(tt)

plt.scatter(tsne_lda[:, 0],tsne_lda[:, 1],color =colormap[lda_keys][:num_example])
#%%
new = LDA_model.components_ / LDA_model.components_.sum(axis=1)[:, np.newaxis]

import numpy as np

x = np.arange(256)

plt.figure(figsize=(10,5))
plt.ylim(0.0001,0.05)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.bar(x, new[0,:])
plt.xlabel('feature_index', fontsize=15)
plt.ylabel('probability', fontsize=15)
plt.savefig("topic1.pdf")
plt.show()
#%%
data2012_data,data2012_label = read_generate_data('2012area','2012image',1536,2048)
data2012_raw_data,data2012_raw_label = read_generate_data('2012raw_data','2012image',1536,2048)
#%%
data2012_data = np.vstack((data2012_data,data2012_raw_data))
data2012_label = np.hstack((data2012_label,data2012_raw_label))
#%%
def mean_category(data,label,category):
    h = np.where(label==category)
    g = data[h,:,:,:]
    g = g[0]
    f = np.mean(g, axis=0)
    f = np.uint8(f)
    return f

#%%find the corresponding label of different clustering
value = np.zeros([6,5])
mean_label = np.zeros(5)
for j in range(6):
    mean_label = mean_category(data2012_data,data2012_label,j)
    mean_label = mean_label.reshape(1,30,30,3)/255
    mean_cnn = cnn_layer_model.predict(mean_label,verbose=2)
    mean_reduced = pca.transform(mean_cnn)
    for i in range(5):
        topic = new[i]
        topic = topic.reshape(1,256)
        f = topic@code_book
        value[j,i] = ((f - mean_reduced)**2).mean(axis=1)

#%% test
image_path = ("./2012image/201208172_T-12-58-58_Dive_01_041.jpg")
#image_files  = glob.glob("./test_image/*")

test_image = imread(image_path)

def lda_fun(x,y):
    m = test_image[x:x+30,y:y+30]
    m = m.reshape(2700)
    return m

def get_feature(stride):
    check_palindrome = np.frompyfunc(lda_fun, 2, 1)
    x = np.arange(0,test_image.shape[0] - 30+1, stride)
    y = np.arange(0,test_image.shape[1] - 30+1, stride)
    X,Y = np.meshgrid(x, y)
    zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
    fs = np.concatenate(zs,axis=0).astype(np.uint8)
    test_all_patches = fs.reshape(-1,30,30,3)/255 #(84924,30,30,3)
    cnn_feature = cnn_layer_model.predict(test_all_patches,verbose=1) #(84924,512)
    feature_reduced = pca.transform(cnn_feature) #(84924,28)
    return feature_reduced

stride = 6
check_palindrome = np.frompyfunc(lda_fun, 2, 1)
x = np.arange(0,test_image.shape[0] - 30+1, stride)
y = np.arange(0,test_image.shape[1] - 30+1, stride)
X,Y = np.meshgrid(x, y)
zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
fs = np.concatenate(zs,axis=0).astype(np.uint8)
test_all_patches = fs.reshape(-1,30,30,3) #(84924,30,30,3)
cnn_feature = cnn_layer_model.predict(test_all_patches,verbose=1) #(84924,512)
feature_reduced = pca.transform(cnn_feature) #(84924,28)
#%%
#code_center = kmeans.predict(feature_reduced)
#g = new[:,code_center]
#category = np.argmax(g,axis=0)
##%%
#color_vector = np.zeros([5,3],dtype=np.uint8)
#color_vector[0] = np.array((0,0,255))
#color_vector[1] = np.array((105,105,105))
#color_vector[3] = np.array((255,0,0))#255 0 0
#color_vector[2] = np.array((0,255,0))#0 255 0
#color_vector[4] = np.array((255,255,0))#255 255 0
##%%
#from skimage import color
#y_pred = category.reshape(252,337)#252,337  
#array_y_pred = y_pred.repeat(6, axis = 0).repeat(6, axis = 1)
#result=color.label2rgb(array_y_pred,colors =color_vector, kind = 'overlay')
#result = np.uint8(result)
#plt.imshow(result)
#%% plot the historgram of the higher probability
high_prob = np.zeros(10)
hi = new[3]
n_top_words = 10
word_idx = np.argsort(new[3])[::-1][:n_top_words]
word_inverse = word_idx[::-1]
for j in range(10):
    index = word_inverse[j]
    high_prob[j] = hi[index]

high_list = list(high_prob)
#f, ax= plt.subplots(5, 1, figsize=(10, 15), sharex=True)
plt.figure(figsize=(8,8))
plt.barh(range(10), high_list, height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
plt.yticks(range(10), [str(word_inverse[n]) for n in range(10)])
plt.xlim(0.001,0.05)
plt.xlabel("probability of feature")
plt.ylabel("feature_index")
plt.title("category_feature distr")
plt.show()
#%%select the higher probability of the features
fea_redu = get_feature(30)
highest_feature = code_book[int(word_idx[0])]
g_feature = ((fea_redu-highest_feature)**2).mean(axis=1)
#%%
import heapq
g = list(g_feature)
re1 = heapq.nsmallest(100, g)
re2 = map(g.index, heapq.nsmallest(100, g))
a = list(re2)
a = np.array(a)
#%%
index_x = a//68
index_y = a%68
#%%
for i in range(100):
    plt.scatter(index_y[i]*30,index_x[i]*30,c = 'b',marker = 'x')
plt.imshow(test_image)
#plt.savefig('test.pdf')



#%% test for singal image
def lda_fun(x,y):
    m = test_image[x:x+30,y:y+30]
    m = m.reshape(2700)
    return m

def get_feature(stride,test_image):
    check_palindrome = np.frompyfunc(lda_fun, 2, 1)
    x = np.arange(0,test_image.shape[0] - 30+1, stride)
    y = np.arange(0,test_image.shape[1] - 30+1, stride)
    X,Y = np.meshgrid(x, y)
    zs = check_palindrome(np.ravel(X.T), np.ravel(Y.T))
    fs = np.concatenate(zs,axis=0).astype(np.uint8)
    test_all_patches = fs.reshape(-1,30,30,3)/255 #(84924,30,30,3)
    cnn_feature = cnn_layer_model.predict(test_all_patches,verbose=2) #(84924,512)
    feature_reduced = pca.transform(cnn_feature) #(84924,28)
    return feature_reduced


#%%
word_idx = np.argsort(new[0])[::-1][:n_top_words]
import heapq
raw_image_files = glob.glob('./all_image_2012/*.jpg')
count = 0
for image_path in raw_image_files:

    start_time = time.time()
    name = image_path.split('\\')[1]
    name_image = name.split('.')[0]
    test_image = imread(image_path)
    hi = new[0]
    n_top_words = 20
    fea_redu = get_feature(30,test_image)
    highest_feature = code_book[int(word_idx[0])]
    g_feature = ((fea_redu-highest_feature)**2).mean(axis=1)
    g = list(g_feature)
    re1 = heapq.nsmallest(50, g)
    re2 = map(g.index, heapq.nsmallest(50, g))
    a = list(re2)
    a = np.array(a)
    index_x = a//68
    index_y = a%68
    for i in range(50):
        plt.scatter(index_y[i]*30,index_x[i]*30,c = 'b',marker = 'x')
    plt.imshow(test_image)
    fig = plt.gcf()
    fig.set_size_inches(2040/100.0/3.0, 1530/100.0/3.0)  
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    #the address to save the result
    plt.savefig('./raw_find_labels/'+str(name_image)+'_point.pdf', bbox_inches='tight')
    plt.close()
    count = count + 1
    print(("---image %d finished in %s seconds ---" % (count,(time.time()-start_time))))
    

#%%
word_idx = np.argsort(new[1])[::-1][:1]
image_path = ("./2012image/201208172_T-12-58-58_Dive_01_041.jpg")
name = image_path.split('/')[2]
name_image = name.split('.')[0]
test_image = imread(image_path)
fea_redu = get_feature(30,test_image)
highest_feature = code_book[int(word_idx)]
g_feature = ((fea_redu-highest_feature)**2).mean(axis=1)
g = list(g_feature)
re2 = map(g.index, heapq.nsmallest(100, g))
a = list(re2)
a = np.array(a)
index_x = a//68
index_y = a%68
for i in range(0,100):
    plt.scatter(index_y[i]*30,index_x[i]*30,c = 'b',marker = 'x')
plt.imshow(test_image)

#%%
def save_labels(name_image,label):
    with open(str('New folder/')+name_image+str('.txt'), 'w+') as newtxtfile:
        newtxtfile.write(name_image+str('\n'))
        newtxtfile.write('x,y,label\n')
        for i in range(50):
            x = index_y[i]*30
            y = index_x[i]*30
            label = int(3)
            newtxtfile.write('{0},{1},{2}\n'.format(x,y,label))
        newtxtfile.close()



#%% 
def get_point_label(name_image,index,color,true_label):
    word_idx = np.argsort(new[index])[::-1][:1]
    test_image = imread(image_path)
    fea_redu = get_feature(30,test_image)
    highest_feature = code_book[int(word_idx[0])]
    g_feature = ((fea_redu-highest_feature)**2).mean(axis=1)
    g = list(g_feature)
    re2 = map(g.index, heapq.nsmallest(50, g))
    a = list(re2)
    a = np.array(a)
    index_x = (a//68)*30
    index_y = (a%68)*30
    for i in range(50):
        plt.scatter(index_y[i],index_x[i],c = str(color),marker = 'x')
        x = index_y[i]
        y = index_x[i]
        label = int(true_label)
        newtxtfile.write('{0},{1},{2}\n'.format(x,y,label))
    plt.imshow(test_image)
    #return index_y, index_x
#    fig = plt.gcf()
#    fig.set_size_inches(2040/100.0/3.0, 1530/100.0/3.0)  
#    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
#    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
#    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
#    plt.margins(0,0)
#    #the address to save the result
#    plt.savefig('./raw_find_labels/'+str(name_image)+'_point.pdf', bbox_inches='tight')
#    plt.close()


#%%
raw_image_files = glob.glob('./all_image_2012/*.jpg')
count = 0
for image_path in raw_image_files:

    start_time = time.time()
    name = image_path.split('\\')[1]
    name_image = name.split('.')[0]
    test_image = imread(image_path)
    start_time = time.time()
    with open(str('New folder/')+name_image+str('.txt'), 'w+') as newtxtfile:
        newtxtfile.write(name_image+str('\n'))
        newtxtfile.write('x,y,label\n')
        get_point_label(name_image,0,'b',0)
        get_point_label(name_image,4,'r',3)
        get_point_label(name_image,3,'g',4)
        newtxtfile.close()
    
    fig = plt.gcf()
    fig.set_size_inches(2040/100.0/3.0, 1530/100.0/3.0)  
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    plt.savefig('./raw_find_labels/'+str(name_image)+'_point.pdf', bbox_inches='tight')
    plt.close()
    count = count + 1
    print(("---image %d finished in %s seconds ---" % (count,(time.time()-start_time))))



#%%
data = np.load('MI_cifar_10_7_29.npz')
data_item = data.items()
name_list = ['a','b','c']
for i in range(3):
    data_compoent = data_item[i][1]
    np.save(str(name_list[i])+'.npy', data_compoent)
