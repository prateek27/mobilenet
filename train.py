from config import *
from model import *
from data_augmentation_generator import *
import os 
#from keras.applications import MobileNet

#model = MobileNet(input_shape=(*INPUT_IMG_SIZE,N_CHANNELS), alpha=1.0, depth_multiplier=1, dropout=0.2, include_top=True, weights=None, 

model = MobileNet(input_shape=(*INPUT_IMG_SIZE,N_CHANNELS), width_multiplier=1.0, depth_multiplier=1, dropout=0.25, include_top=True, weights=None,classes=NO_OF_CLASSES)

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
model.summary()


#Create Checkpoint Directory
if(not os.path.isdir(SAVED_WEIGHTS_DIR)):
   os.mkdir(SAVED_WEIGHTS_DIR)

#Model Checkpoint
ckpt = ModelCheckpoint(filepath='weights/{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', save_best_only=True)
tb = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=1 )
  
   
   
# Training Model
model.fit_generator(train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    steps_per_epoch = STEPS_PER_EPOCH,
                    workers=6, epochs = 50, callbacks = [ckpt,tb])

 