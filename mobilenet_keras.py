

from datagenerator import *

from keras.callbacks import ModelCheckpoint, TensorBoard


train_generator = DataGenerator(batch_size=128,dim=(224,224),n_channels=3,list_IDs=partition['train'],
                                labels=labels,n_classes=NO_OF_CLASSES)

val_generator = DataGenerator(batch_size=128,dim=(224,224),n_channels=3,list_IDs=partition['validation'],
                                labels=labels,n_classes=NO_OF_CLASSES)

ckpt = ModelCheckpoint(filepath='weights_keras/{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', save_best_only=True)
tb = TensorBoard(log_dir='./logs_keras', histogram_freq=1, batch_size=128 )


from keras.applications import MobileNetV2
from keras.applications import MobileNet

k_model = MobileNet(input_shape=(224,224,3), alpha=1.0, depth_multiplier=1, 
                              dropout=1e-3, include_top=True, weights=None, 
                              input_tensor=None, pooling=None, classes=200)


k_model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

k_model.summary()

k_model.fit_generator(train_generator, validation_data=val_generator,epochs=35,callbacks=[ckpt],workers=6,use_multiprocessing=True)

