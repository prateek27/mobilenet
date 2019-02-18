
# coding: utf-8

# In[60]:


import numpy as np
# import keras
# from keras.preprocessing import image
from PIL import Image
import os


# In[74]:


# class DataGenerator(keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
#                  n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)

#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             # Store sample
# #             img = image.load_img(ID,target_size=self.dim)
# #             img = image.img_to_array(img)

#             img = Image.open(ID)
#             img = img.convert('RGB')
#             img = img.resize(self.dim)
#             X[i,] = np.array(img, dtype=np.float32)/255.0
#             img.close()
            
#             # Store class
#             y[i] = self.labels[ID]

#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[75]:


partition = {
    'train': [],
    'validation': []
}
labels = {}
class_ids = {}
cnt = 0


# In[76]:

X_train = []
Y_train = []
X_test = []
Y_test = []

base_path = '/datadrive/tiny-imagenet-200/train'

def img_load( path, dim = (224,224) ):
    img = Image.open(ID)
    img = img.convert('RGB')
    img = img.resize(dim)
    img = np.array(img, dtype=np.float32)/255.0
    return img

# In[77]:


print ("Process Training Data")


# In[78]:


for label in os.listdir(base_path):
    temp = os.path.join(base_path, label)
    
    if class_ids.get(label) is None:
        class_ids[label] = cnt
        cnt += 1
        print ( cnt , "Done" )
    
    img_fldr_path = os.path.join(temp, 'images')
    for imgs in os.listdir(img_fldr_path):
        ID = os.path.join(img_fldr_path, imgs)
        #partition['train'].append( ID )
        #labels[ID] = class_ids[label]
        X_train.append( img_load(ID) )
        Y_train.append( class_ids[label] )
        
    if cnt == 150:
        break


# In[79]:

print ("Training data generated")

X_train = np.array( X_train )
Y_train = np.array( Y_train )
print ( X_train.shape, Y_train.shape )

np.save('./data/x_train.npy', X_train )
np.save('./data/y_train.npy', Y_train )

#print(len(labels))


# In[80]:


NO_OF_CLASSES = len((os.listdir(base_path)))


# In[81]:


print ("Process Validation Data")
base_path_valid = '../tiny-imagenet-200/val'
st = '../tiny-imagenet-200/val/images/'


with open(os.path.join(base_path_valid,"val_annotations.txt")) as f:
    
    lines = f.readlines()
    for line in lines:
        tokens = line.split()
        img_name = tokens[0]
        img_label = tokens[1]
        ID = os.path.join(st,img_name)
#         partition['validation'].append(ID)
#         labels[ID] = class_ids[img_label]
        if ( class_ids.get( img_label ) is None ):
            continue
        X_test.append( img_load(ID) )
        Y_test.append( class_ids[img_label] )


# In[82]:


X_test = np.array( X_test )
Y_test = np.array( Y_test )

print ( X_test.shape, Y_test.shape )
np.save('./data/x_test.npy', X_test )
np.save('./data/y_test.npy', Y_test )

"""
train_generator = DataGenerator(batch_size=128,dim=(227,227),n_channels=3,list_IDs=partition['train'],
                                labels=labels,n_classes=NO_OF_CLASSES)

val_generator = DataGenerator(batch_size=128,dim=(227,227),n_channels=3,list_IDs=partition['train'],
                                labels=labels,n_classes=NO_OF_CLASSES)
"""


# In[83]:


