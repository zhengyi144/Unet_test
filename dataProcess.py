from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import cv2
from keras.models import Model,load_model
from keras.layers import Conv2D,MaxPooling2D,Dropout,Concatenate,Input,UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt


Sky = [128,128,128]
Building = [255,255,255]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def trainGenerator(batch_size,train_path,image_folder,mask_folder,\
    aug_dict,image_color_mode='grayscale',mask_color_mode='grayscale',image_save_prefix='image',\
    mask_save_prefix='mask',flag_multi_class=False,num_class=2,save_to_dir=None,
    target_size=(256,256),seed=1):
    image_datagen=ImageDataGenerator(**aug_dict)
    mask_datagen=ImageDataGenerator(**aug_dict)

    image_generator=image_datagen.flow_from_directory(train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    
    mask_generator=mask_datagen.flow_from_directory(train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    #print(mask_generator)

    train_generator=zip(image_generator,mask_generator)
    """
    生成图像
    for i in range(60):
        mask_generator.next()
        image_generator.next()
    """

    #
    for (img,mask) in train_generator:
        img,mask=adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
    
def adjustData(img,mask,flag_multi_class,num_class):
    #print(np.shape(img))
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def Unet(pretrained_weights=None,input_size=(256,256,1)):
    input=Input(input_size)
    conv1=Conv2D(32,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(input)
    conv2=Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    pool1=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv2)  #(128,128)
    conv3=Conv2D(64,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv4=Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv3)
    pool2=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv4)   #(64,64)
    conv5=Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv6=Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv5)
    pool3=MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv6)    #(32,32)
    conv7=Conv2D(256,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv8=Conv2D(512,(3,3),activation='relu',padding='same',kernel_initializer='he_normal')(conv7)
    drop1=Dropout(0.5)(conv8)    
    pool4=MaxPooling2D(pool_size=(2,2))(drop1)   #(16,16)

    conv9 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv10 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    drop2 = Dropout(0.5)(conv10)
    
    up1=UpSampling2D(size=(2,2))(drop2)  #(32,32)
    up1_conv1=Conv2D(512,2,activation='relu',padding='same',kernel_initializer = 'he_normal')(up1)
    merge1=Concatenate()([drop1,up1_conv1])  
    merge1_conv1=Conv2D(512,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge1)
    merge1_conv2=Conv2D(256,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge1_conv1)

    up2=UpSampling2D(size=(2,2))(merge1_conv2) #(64,64)
    up2_conv1=Conv2D(256,2,activation='relu',padding='same',kernel_initializer = 'he_normal')(up2)
    merge2=Concatenate()([conv6,up2_conv1])
    merge2_conv1=Conv2D(256,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge2)
    merge2_conv2=Conv2D(128,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge2_conv1)

    up3=UpSampling2D(size=(2,2))(merge2_conv2)  #(128,128)
    up3_conv1=Conv2D(128,2,activation='relu',padding='same',kernel_initializer = 'he_normal')(up3)
    merge3=Concatenate()([conv4,up3_conv1])
    merge3_conv1=Conv2D(128,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge3)
    merge3_conv2=Conv2D(64,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge3_conv1)

    up4=UpSampling2D(size=(2,2))(merge3_conv2)  #(256,256)
    up4_conv1=Conv2D(64,2,activation='relu',padding='same',kernel_initializer = 'he_normal')(up4)
    merge4=Concatenate()([conv2,up4_conv1])
    merge4_conv1=Conv2D(64,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge4)
    merge4_conv2=Conv2D(32,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge4_conv1)
    merge4_conv3=Conv2D(2,3,activation='relu',padding='same',kernel_initializer = 'he_normal')(merge4_conv2)
    out=Conv2D(1,1,activation='sigmoid')(merge4_conv3)

    model=Model(input=input,output=out)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    
    return model
    
def train(train_path=None,pretrained_weights=None,out_path=None):
    checkpoint_dir='./checkpoint/'
    data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    generator=trainGenerator(4,train_path,'image','label',data_gen_args)
    model=Unet(pretrained_weights)
    model_checkpoint=ModelCheckpoint(checkpoint_dir+'Unet_model-ep{epoch:03d}-loss{loss:.3f}.h5',monitor='loss',period=1,save_weights_only=True,save_best_only=True)
    model.fit_generator(generator,steps_per_epoch=1000,epochs=10,callbacks=[model_checkpoint])

def validation(model_path,data_path,output_path):
    try:
        model=load_model(model_path)
    except:
        print("except load_model")
        model=Unet()
        model.load_weights(model_path)
    
    images_name=os.listdir(data_path)
    np.random.shuffle(images_name)
    for image_name in images_name:
        image_path=os.path.join(data_path,image_name)
        image=cv2.imread(image_path,1)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 
        image=cv2.resize(image,(256,256))
        image=np.asarray(image)/255
        image=np.reshape(image,[-1,256, 256, 1])
        
        #构建输入输出函数
        out_layer=K.function([model.layers[0].input],[model.layers[32].output])
        #将image作为输入
        f1=out_layer([image])[0]
        
        #print(np.shape(f1))
        
        for index in range(f1.shape[-1]):
            filter_img=f1[:,:,:,index]   #(1, 256, 256)
            #print(np.shape(filter_img))
            filter_img.shape=filter_img[0,:,:].shape
            plt.subplot(8,8,index+1)
            plt.imshow(filter_img)
        plt.show()
        break


        #out=model.predict(image)
        """img_out = np.zeros((256,256) + (3,))
        print(np.shape(img_out))
        for i in range(2):
            img_out[img >0.8,:] = COLOR_DICT[i]
        img_out / 255"""

        #out_path=os.path.join(output_path,'layer1_filter'+image_name)
        #cv2.imwrite(out_path,img)
        
    print("validation finish!")

if  __name__=="__main__":
    data_path=r'D:\DeepLearning\unet-master\data\membrane\train'
    out_path=r'D:\DeepLearning\unet-master\data\membrane\train\aug2'
    validation_path=r'D:\DeepLearning\unet-master\data\membrane\test'
    model_path='./checkpoint/Unet_model-ep006-loss0.157.h5'
    #train(train_path=data_path)
    validation(model_path,validation_path,validation_path)
    