
from datagenerator import *



"""
Paper link - https://arxiv.org/pdf/1704.04861.pdf
"""
from keras.layers import ZeroPadding2D,Conv2D,BatchNormalization,Input,Dropout,DepthwiseConv2D,Input
from keras.layers import ReLU,GlobalAveragePooling2D,GlobalMaxPool2D,Reshape,Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard

def MobileNet(
    input_shape=(224,224,3),
    width_multiplier = 1.0, #changes number of filters
    depth_multiplier = 1, # Resolution Multiplier
    include_top = True,
    weights = None,
    dropout = 1e-3,
    input_tensor = None,
    pooling = None, #Global Average/Max Pooling or None
    classes = 1000,
    ):

    #Input Shape
    if input_shape is None:
        default_size = 224
    else:
        rows,cols = input_shape[0],input_shape[1]
        #Make sure we use one of the mentioned sizes
        if rows==cols and rows in [128,160,192,224]:
            default_size = rows
        else:
            default_size = 224

    #Input Tensor
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor 

    #Standard Convolution
    #Block 0
    x = conv_block(img_input,32,width_multiplier,strides=(2,2))

    #Block 1
    x = depthwise_conv_block(x,64,width_multiplier,depth_multiplier,block_id=1)
    #Block2
    x = depthwise_conv_block(x,128,width_multiplier,depth_multiplier,strides=(2,2),block_id=2)
    #Block 3
    x = depthwise_conv_block(x,128,width_multiplier,depth_multiplier,block_id=3)
    #Block 4
    x = depthwise_conv_block(x,256,width_multiplier,depth_multiplier,strides=(2,2),block_id=4)
    #Block 5
    x = depthwise_conv_block(x,256,width_multiplier,depth_multiplier,block_id=5)
    #Block 6
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,strides=(2,2),block_id=6)
    #Block 7
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,block_id=7)
    #Block 8
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,block_id=8)
    #Block 9
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,block_id=9)
    #Block 10
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,block_id=10)
    #Block 11
    x = depthwise_conv_block(x,512,width_multiplier,depth_multiplier,block_id=11)
    #Block 12
    x = depthwise_conv_block(x,1024,width_multiplier,depth_multiplier,strides=(2,2),block_id=12)
    #Block 13
    x = depthwise_conv_block(x,1024,width_multiplier,depth_multiplier,block_id=13)

    if include_top:
        shape = (1,1,int(1024*width_multiplier))
        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape,name='reshape_1')(x)
        x = Dropout(dropout,name='dropout')(x)
        x = Conv2D(classes,(1,1),padding='same',name='conv_fc')(x)
        x = Activation('softmax',name='softmax')(x)
        x = Reshape((classes,),name='reshape_2')(x)
    else:
        if pooling=='avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)

    #Create Model using functional API
    model = Model(inputs=img_input,outputs=x,name='mobilenet')

    if weights is not None:
        model.load_weights(weights)

    return model



def conv_block(inputs,filters,width_multiplier,kernel_size=(3,3),strides=(1,1)):
    #Inital Conv Layer with Batch Norm and Relu6
    #Input Shape : 4D Tensor (Samples,Rows,Cols,Channels)
    #Output Shape : 4D Tensor(Samples,New_Rows,New_Cols,Channels)
    #Width_Multiplier : Changes Number of Filters

    filters = int(filters*width_multiplier)

    #Params in Zero Padding ((top_pad, bottom_pad), (left_pad, right_pad))
    x = ZeroPadding2D(padding=((0,1),(0,1)),name='conv1_pad')(inputs)

    #Apply Standard Convolution without Bias and Batch Norm
    x = Conv2D(filters,kernel_size,padding='valid',use_bias=False,strides=strides,name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)


    # ReLu6 = min(max(features, 0), 6)
    x = ReLU(6.,name='conv1_relu')(x)
    return x


def depthwise_conv_block(inputs,pointwise_conv_filters,width_multiplier,depth_multiplier=1,strides=(1,1),block_id=1):

    """
    depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
    The total number of depthwise convolution output
    channels will be equal to `filters_in * depth_multiplier`.
    """

    #Update the Number of Output Filters
    pointwise_conv_filters = int(pointwise_conv_filters*width_multiplier)

    if strides==(1,1):
        x = inputs
    else:
        x = ZeroPadding2D(padding=((0,1),(0,1)),name='conv_pad_%d'%block_id)(inputs)

    # Depth Wise Convolution
    x = DepthwiseConv2D((3,3),padding='same' if strides==(1,1) else 'valid',depth_multiplier=depth_multiplier,strides=strides,use_bias=False,name='conv_dw_%d'%block_id)(x)
    x = BatchNormalization(name='conv_dw_%d_bn'%block_id)(x)
    x = ReLU(6.,name='conv_dw_%d_relu'%block_id)(x)

    # PointWise Convolution with 1X1 Filters, No of Filters = pointwise_conv_filters	
    x = Conv2D(pointwise_conv_filters,(1,1),padding='same',use_bias=False,strides=(1,1),name='conv_pw_%d'%block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn'%block_id)(x)
    x = ReLU(6.,name='conv_pw_%d_relu'%block_id)(x)

    return x


# In[3]:


model = MobileNet(width_multiplier=1,depth_multiplier=1,classes=NO_OF_CLASSES)

model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')

model.summary()






train_generator = DataGenerator(batch_size=128,dim=(224,224),n_channels=3,list_IDs=partition['train'],
                                labels=labels,n_classes=NO_OF_CLASSES)

val_generator = DataGenerator(batch_size=128,dim=(224,224),n_channels=3,list_IDs=partition['train'],
                                labels=labels,n_classes=NO_OF_CLASSES)



ckpt = ModelCheckpoint(filepath='weights/{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', save_best_only=True)
tb = TensorBoard(log_dir='./logs', histogram_freq=1, batch_size=32 )





model.fit_generator(train_generator,
                    validation_data=val_generator,
                    use_multiprocessing=True,
                    workers=6, epochs = 100, callbacks = [ckpt,tb ])
