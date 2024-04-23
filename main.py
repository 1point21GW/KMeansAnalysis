# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd

path = 'C:/Users/sgonilov/OneDrive - Intel Corporation/Desktop/Klarity - 022224/HiRes/Off Center'
# change the working directory to the path where the images are located
os.chdir(path)

# this list holds all the image filename
defects = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
          # adds only the image files to the flowers list
            defects.append(file.name)

print('Done!')

# load the image as a 224x224 array
img = load_img(defects[0], target_size=(224,224))
# convert from 'PIL.Image.Image' to numpy array
img = np.array(img)

reshaped_img = img.reshape(1,224,224,3)

x = preprocess_input(reshaped_img)
print('Images preprocessed')

# load the model first and pass as an argument
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

data = {}
# lop through each image in the dataset
for defect in defects:
    feat = extract_features(defect,model)
    data[defect] = feat
          
 
# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))
feat.shape

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
feat.shape

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components= 200, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

print('PCA done')

pca.explained_variance_ratio_.cumsum()

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.plot(60, 0.85, 'o')
plt.axhline(y = 0.85, xmax = 0.31, color = 'gray', linestyle = '--') 
plt.axvline(x = 60, ymin = 0, ymax = 0.83, color = 'gray', linestyle = '--')
plt.text(61, 0.83, f"({60}, {0.85})", fontsize=8)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.show()

plt.scatter(
    x[:, 0], x[:, 1], marker=".", s=30, lw=0, alpha=0.7, edgecolor="k"
)
plt.title("Similarity Among Defects")
plt.xlabel("Feature space for the 1st feature")
plt.ylabel("Feature space for the 2nd feature")

# this is just incase you want to see which value for k might be the best 
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

import warnings
warnings.filterwarnings("ignore")

sse = []
sih = []
db = []
ch = []
list_k = list(range(2, 10))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    
    sse.append(km.inertia_)
    sih.append(silhouette_score(x, km.labels_))
    db.append(davies_bouldin_score(x, km.labels_))
    ch.append(calinski_harabasz_score(x, km.labels_))

print('K means done')

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
plt.title("Clusters vs Inertia")
plt.axvline(x = 3, ymin = 0, ymax = 0.57, color = 'gray', linestyle = '--')
plt.axvline(x = 5, ymin = 0, ymax = 0.285, color = 'gray', linestyle = '--')
plt.axvline(x = 7, ymin = 0, ymax = 0.14, color = 'gray', linestyle = '--')

ax = plt.figure(figsize=(6, 6)).gca()
plt.plot(list_k, sih, 'o')
plt.axhline(y = 1, color = 'r', linestyle = '-') 
plt.title("Clusters vs Silhouette Score")
plt.xlabel("No. of Clusters")
plt.ylabel("Silhouette score")
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(list_k, db, 'o')
plt.axhline(y = 0, color = 'r', linestyle = '-') 
plt.title("Clusters vs Davies-Bouldin Index")
plt.xlabel("No. of Clusters")
plt.ylabel("DB score")
plt.grid()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(list_k, ch, 'o')
plt.title("Clusters vs Calinski-Harabasz Index")
plt.xlabel("No. of Clusters")
plt.ylabel("CH score")
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=3, random_state=22).fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

  # function that lets you view a cluster (based on identifier)      

def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    plt.title('Cluster #' + str(cluster+1), fontsize = 40)
    plt.axis('off')
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 50 images to be shown at a time
    if len(files) > 50:
        print(f"Clipping cluster size from {len(files)} to 50")
        files = files[:50]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')

for i in groups:
    view_cluster(i)
