from config import *
from keras.preprocessing.image import ImageDataGenerator


train_image_gen = ImageDataGenerator(
    rescale= 1/255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.3,
    horizontal_flip = True,
)
val_image_gen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_image_gen.flow_from_directory(
    TRAIN_BASE_PATH,
    target_size =(INPUT_IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)

val_generator = val_image_gen.flow_from_directory(
    VAL_BASE_PATH,
    target_size =(INPUT_IMG_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = 'categorical'
)








