
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_hub as hub
import keras
from skimage import io
import skimage
import matplotlib.pyplot as plt
import numpy as np
import functions


# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4")
# ])
# m.build([None, 224, 224, 3])  # Batch input shape.

# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", output_shape=[1280],
#                    trainable=False),  # Can be True, see below.
# ])
# m.build([None, 224, 224, 3])  # Batch input shape.

# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4",
#                    trainable=False),  # Can be True, see below.
# ])
# m.build([None, 299, 299, 3])  # Batch input shape.
# 

# m = tf.keras.Sequential([
#     hub.KerasLayer("imagenet_inception_v3_feature_vector_4/",
#                    trainable=False),  # Can be True, see below.
# ])
# m.build([None, 299, 299, 3])  # Batch input shape.
# 

# m = tf.keras.Sequential([
#     hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/4",
#                    trainable=False),  # Can be True, see below.
# ])
# m.build([None, 96, 96, 3])  # Batch input shape.

# In[2]:


m = tf.keras.Sequential([
    hub.KerasLayer("experts_bit_r50x1_in21k_substance_1/",
                   trainable=False),  # Can be True, see below.
])
m.build([None, 299, 299, 3])  # Batch input shape.


# In[3]:


print("About model: ")
print(m.summary())


# In[4]:


images = io.imread_collection("data/RGB/Resized_299/*.jpeg")


# In[5]:


#print(images)


# In[6]:


image_set = []
for image in images:
    image_set.append(image.astype("float32"))


# In[7]:


image_set = np.array(image_set)


# In[8]:


print("Image set shape = {}".format(image_set.shape))


# # Important Details
# - Float32 type images
# - Batch have to be np.array

# ---

# ### Some Euclidian Distance Application

# In[9]:


predictions = m.predict(image_set)


# In[10]:


print("Prediction shape = {}".format(predictions.shape))


# ----

# # PCA Application

# In[11]:


from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# In[12]:


X = predictions.copy()


# In[13]:


pca = PCA(n_components=2)
pca.fit(X)


# In[14]:


print("PCA variance ration = {}".format(pca.explained_variance_ratio_))
#print(pca.singular_values_)


# In[15]:


X_pca = pca.transform(X)


# In[16]:


X_pca.shape


# fig = plt.figure(1, figsize=(10,10))
# ax = Axes3D(fig)
# ax.scatter3D(X_pca[:,0], X_pca[:,1], X_pca[:,2])
# plt.show()

# In[17]:


plt.scatter(X_pca[:,0], X_pca[:,1])


# # Dİmension Reduced Data for Clustering

# In[92]:


from sklearn.decomposition import PCA


# In[93]:


X = predictions.copy()


# In[94]:


pca = PCA(n_components=10)
pca.fit(X)


# In[95]:


print("PCA variance ratio = {}".format(pca.explained_variance_ratio_))
#print(pca.singular_values_)


# In[96]:


X_cluster_data = pca.transform(X)


# In[97]:


print("Dimension Reduced Data shape = {}".format(X_cluster_data.shape))


# -----

# ### Let's use StandartScalar to improve PCA ( It is not useful.)

# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# scaler.fit(X)
# X_scaled = scaler.transform(X)

# print(X_scaled.shape)

# scaled_pca = PCA(n_components=2)
# scaled_pca.fit(X_scaled)

# print(scaled_pca.explained_variance_ratio_)
# print(scaled_pca.singular_values_)

# X_scaled_pca = scaled_pca.transform(X)

# plt.scatter(X_scaled[:,0], X_scaled[:,1])

# ### Let's use non-linear dimension reduction 

# from sklearn.decomposition import KernelPCA

# transformer = KernelPCA(n_components=2, kernel="cosine")
# transformer.fit(X)
# X_transformed = transformer.transform(X)

# X_transformed.shape

# plt.scatter(X_transformed[:,0], X_transformed[:,1])

# # Clustering

# In[98]:


from sklearn.cluster import KMeans


# In[99]:


active_cluster_number = 3
print("Active Cluster Number = {}".format(active_cluster_number))


# In[100]:


kmeans = KMeans(n_clusters=active_cluster_number)
kmeans.fit(X_cluster_data)


# In[101]:


label = kmeans.labels_
print(label)


# f , ax = plt.subplots()
# ax.scatter(X_pca[np.where(label == 0)][:,0], X_pca[np.where(label == 0)][:,1], edgecolors="r")
# ax.scatter(X_pca[np.where(label == 1)][:,0], X_pca[np.where(label == 1)][:,1], cmap="gray")
# ax.scatter(X_pca[np.where(label == 2)][:,0], X_pca[np.where(label == 2)][:,1], cmap="gray")

# In[28]:


#kmeans.cluster_centers_.shape


# In[102]:


dist = kmeans.transform(X_cluster_data)


# In[103]:


min_dist = []
for ele in dist:
    min_dist.append(min(ele))


# In[104]:


min_dist.sort()
#print(min_dist)
length = min_dist[-5]
print("Estimated Circular Distance = {}".format(length))


# - We can say that, length have to be smaller than 13.

# In[105]:


def IsMemberofCluster(feature_data, kmeans = kmeans):
    '''
    It checks, new data is member of any cluster or not?
    '''
    row,col = feature_data.shape
    
    # It begin with false. If it is member of any cluster, it will return true.
    imc = []
        
    distances = kmeans.transform(feature_data)
    #print(distances)
    
    for vector in distances:
        if np.all(vector > length):
            imc.append(False)
            continue
        imc.append(True)
        
    return imc


# -----

# imgs = io.imread_collection("test/test299/image_0.jpeg")

# print(imgs)

# test_imgs = []
# for img in imgs:
#     test_imgs.append(img.astype("float32"))

# test_imgs = np.array(test_imgs)
# test_imgs.shape

# test_features = m.predict(test_imgs)

# test_features[-1]

# pca.explained_variance_ratio_

# test_feature_data = pca.transform(test_features)

# print(test_features[-1])

# pca.explained_variance_ratio_

# label = IsMemberofCluster(test_feature_data)

# print(label)

# -----

# # Let's Add New Cluster

# In[113]:


imag_list = functions.Grab_and_parse()
plt.imshow(imag_list[0])


# In[114]:


### Main Add Cluster###

# let's select new image that is not in any cluster.

#imag = io.imread("test/test299/image_6.jpeg")
imag = imag_list[0]
imag = imag.astype("float32")
imag1 = skimage.transform.rotate(imag,90)
imag2 = skimage.transform.rotate(imag,180)
imag3 = skimage.transform.rotate(imag,270)
# We add all versions of image in the list.
image_list = [imag, imag1, imag2, imag3]


# In[115]:


# Let's prepare list to model
image_list = np.array(image_list)
# Let's use model to extract feature vector
image_features = m.predict(image_list)
#image_features.shape


# In[116]:


# Now let's reduce their dimensions with PCA model.
pca_image_features = pca.transform(image_features)

# Lets check are they member of any cluster?
flags = IsMemberofCluster(pca_image_features, kmeans=kmeans)
print("Cluster flags = {}".format(flags))


# In[121]:


if np.any(flags):
    # We found which ones can be in cluster.
    # Now,  lets find which cluster include this image.
    
    index = flags.index(True)
    print(index)
    
    cluster = kmeans.predict(pca_image_features[index].reshape(1,-1))
    print("This image is in {} cluster".format(cluster))
    
    l = list(kmeans.labels_)
    times = l.count(cluster)
    #print(times)
    if times < 10:
        X = np.concatenate((X,image_features),axis=0)
        print("It is in Cluster but still new!")
        pca = PCA(n_components=10)
        pca.fit(X)
        X_cluster_data = pca.transform(X)
        times = times + 4
        kmeans = KMeans(n_clusters=active_cluster_number)
        kmeans.fit(X_cluster_data)
        


# In[111]:


flags = np.array(flags)
if np.all(flags == False):
    # In this case, we have to think new cluster.
    active_cluster_number = active_cluster_number + 1 
    X = np.concatenate((X,image_features),axis=0)
    #X.shape
    
    # Firstly, update PCA.
    pca = PCA(n_components=10)
    pca.fit(X)
    #print(pca.explained_variance_ratio_)
    #print(pca.singular_values_)
    X_cluster_data = pca.transform(X)
    
    # Fit new kmean.
    kmeans = KMeans(n_clusters=active_cluster_number)
    kmeans.fit(X_cluster_data)
    label = kmeans.labels_
    print("Cluster label = {}".format(label))
    
    
    # Find new cluster


# In[112]:


print("Active Cluster Number = {}".format(active_cluster_number))

