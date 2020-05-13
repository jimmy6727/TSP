# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sat Apr 18 07:22:41 2020

# @author: jimmyjacobson
# """

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import concorde.tsp as concorde
import os
from PIL import Image
from sklearn.model_selection import train_test_split

np.random.seed(1337)  # for reproducibility
import numpy as np
import skopt
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.utils import class_weight
#from keras.utils import to_categorical
import re

def plot_to_vec(plotname):
    
    img = Image.open(plotname).convert('L')
    arr = np.array(img)
    shape=arr.shape
    # print('\nData Type: %s' % arr.dtype)
    # print('Min: %.3f, Max: %.3f' % (arr.min(), arr.max()))
    # print(shape)
    img_r = resize(arr, output_shape=(120,160,1), anti_aliasing=True, preserve_range=True)
    img_r=((img_r-img_r.min()) / (img_r.max()-img_r.min())) * (254)+1
    img_r = img_r / 255
    #new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
    # print('Data Type: %s' % img_r.dtype)
    # print('Min: %.3f, Max: %.3f' % (img_r.min(), img_r.max()))
    # print(str(img_r.shape))
        
    return(img_r, shape)

def plot_folder_to_vectorized_df(foldername):
    files = os.listdir(foldername)
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    outs = []
    labels = []
    for i in range(len(files)):
        if not files[i].startswith('.'):
            p = foldername+'/'+str(files[i])
            vec=plot_to_vec(p)
            outs.append(vec[0])
            if i%2 == 0:
                labels.append(1)
            else:
                labels.append(0)
    return(outs,np.array(labels))

def normalize(a, axis=(1,2)): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

def main():
    # Image data as a two item list of image arrays, labels
    data = plot_folder_to_vectorized_df('MyData/NGAPlots')
        
        # # For debugging: x gets the first image from data, which is label 1
        # x = data[0][1]  # this is a Numpy array with shape (3, 150, 150)
        # #x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        # Image.fromarray((x * 255).astype(np.uint8)).show()
        
    
    # for HPO
    
    # SPACE = [skopt.space.Real(0.00001, 0.001, name='_learning_rate', prior='log-uniform'),
    #           skopt.space.Categorical([16, 32, 64, 128], name='_batch_size'),
    #           skopt.space.Categorical([4,8,16,32,64], name='_conv_layer_size'),
    #           skopt.space.Categorical([4,8,16,32,64, 128], name='_dense_layer_size'),
    #           skopt.space.Real(0.05, .5, name='_dropout'),
    #           skopt.space.Real(0.01, .1, name='_alpha'),
    #           skopt.space.Real(0.01, .1, name='_alpha1')]
    
    
    # @skopt.utils.use_named_args(SPACE)
    
    # def objective(_learning_rate, _dropout, _alpha, _alpha1, _conv_layer_size, _dense_layer_size):
    # def main():
    _conv_layer_size = 16
    
    _dense_layer_size = 64
    
    _learning_rate = 0.0001
    
    _dropout = 0.06
    
    _alpha,_alpha1 = 0.1,0.1
        
    _batch_size=64
    
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(data[0]), np.array(data[1]), test_size=0.5, random_state=0)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.5, random_state=0)
    
    # print("\n\n******** Testing HPO's ********")
    # #print("**batch_size = \t",_batch_size)
    # print("**conv_layer_size =\t",_conv_layer_size)
    # print("**learning_rate = \t",_learning_rate)
    # print("**dropout = \t",_dropout)
    # print("**dense_layer_size = \t",_dense_layer_size)
    # print("**alpha = \t",_alpha)
    # print("**alpha1 = \t",_alpha1)
    # print("**********************************\n")
    
    # Y_train = to_categorical(Y_train)
    # Y_test = to_categorical(Y_test)
    # Y_valid = to_categorical(Y_valid)
    
    # # Check to see the input data looks good
    # # img_r = Image.fromarray(data[0][1].reshape(120,160)).show()
    
    #    #Report pixel means and standard deviations
    print("\n=========== Dataset stats before data augmentation ===========")
    print("Train: ", X_train.shape, round(X_train.mean(),3), round(X_train.std(),3), round(X_train[1].min(), 3),round(X_train[1].max(),3))
    
    # Image augmentation generator object for training 
    datagen = ImageDataGenerator(
            #featurewise_center=True,
            #featurewise_std_normalization=True,
            shear_range = 20,
            #width_shift_range=.1,
            #height_shift_range=.1,
            fill_mode='nearest',
            zoom_range = [1,1.3],
            #rescale=1.0/255.0,
            rotation_range=360,
            horizontal_flip=True,
            vertical_flip=True)
    
    test_datagen=ImageDataGenerator(#featurewise_center=True,
                                    #featurewise_std_normalization=True,
                                    rotation_range=360,
                                    zoom_range = [1.15,1.2],
                                    horizontal_flip=True,
                                    vertical_flip=True)
    
    def gen_with_norm(gen, norm):
        for x, y in gen:
            yield norm(x), y
        

    # calculate and display the mean and std on the training dataset
    print("\n=========== Sample data stats after data augmentation ===========")
   
    f, axarr = plt.subplots(4,4)
    for X_batch, y_batch in gen_with_norm(gen=(datagen.flow(X_train, Y_train, batch_size=25)),norm = normalize):
        for i in range(4):
            for j in range(4):
                x = X_batch[(4*i)+j]  
                x = x.reshape(120,160) 
                axarr[i,j].imshow(x*255)
                axarr[i,j].axis('off')
                axarr[i,j].set_title(str(y_batch[(4*i)+j]))
        f.suptitle("Sample of Labeled Training Data")
        f.show()
        break
    
    f, axarr = plt.subplots(4,4)
    for X_batch, y_batch in gen_with_norm(gen=(test_datagen.flow(X_train, Y_train, batch_size=25)),norm = normalize):
        for i in range(4):
            for j in range(4):
                x = X_batch[(4*i)+j]  
                x = x.reshape(120,160)  
                axarr[i,j].imshow(x*255)
                axarr[i,j].set_title(str(y_batch[(4*i)+j]))
        f.suptitle("Sample of Labeled Testing Data")
        f.show()
        break
    
    train_generator = datagen.flow(
    X_train, Y_train, 
    batch_size=_batch_size)
     

    # Get single batch to test normalization
    for batchX, y_batch in datagen.flow(X_train, Y_train, batch_size=_batch_size):
        break
    print("Sample Train Batch Stats: ", batchX.shape, round(batchX.mean(),3), round(batchX.std(),3), round(batchX[0].min(), 3),round(batchX[0].max(),3))
    
   
    validation_generator = test_datagen.flow(
            X_valid, Y_valid,
            batch_size=_batch_size)
    
    test_generator = test_datagen.flow(
            X_test, Y_test,
            batch_size=_batch_size)
    
       
    for batchX, y_batch in test_datagen.flow(X_train, Y_train, batch_size=_batch_size):
        break
    print("Sample Test Batch Stats: ", batchX.shape, round(batchX.mean(),3), round(batchX.std(),3), round(batchX.min(), 3),round(batchX.max(),3))
        
    print('\n=========\nTrain min=%.3f, max=%.3f' % (X_train.min(), X_train.max()))
    print('Test min=%.3f, max=%.3f' % (X_test.min(), X_test.max()))
    print('=========\n')
    
    
    adam = optimizers.Adam(learning_rate=_learning_rate)
    
    
    model = Sequential()
    model.add(Conv2D(_conv_layer_size, (6, 6),input_shape=(120,160,1)))
    model.add(LeakyReLU(alpha=_alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(_dropout))
    model.add(Conv2D(_conv_layer_size//2, (4, 4)))
    model.add(LeakyReLU(alpha=_alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(_dropout))
    model.add(Conv2D(_conv_layer_size//4, (2, 2)))
    model.add(LeakyReLU(alpha=_alpha))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(_dropout))
    model.add(Flatten()) 
    model.add(Dense(_dense_layer_size))
    model.add(LeakyReLU(alpha=_alpha1))   
    model.add(Dropout(_dropout))
    model.add(Dense(1,activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    
    
    #Mclass_weights = class_weight.compute_class_weight('balanced',
                                                      # np.unique(Y_train),
    # Big class weights are a quick and dirty 
    # hack to remove false positives                                                 # Y_train)
    Mclass_weights = [50,1]
    
    model.fit_generator(
            train_generator,
            steps_per_epoch= 10,
            epochs=30,
            verbose=1,
            class_weight = Mclass_weights,
            validation_data=validation_generator,
            validation_steps= 10)
    model.save('TSPplotmodel1.h5')  # always save your weights after training or during training
    
    time0 = time.time()
    pred = model.predict_generator(test_generator, verbose=1,steps=1)
    f, axarr = plt.subplots(4,4)
    for X_batch, y_batch in test_datagen.flow(X_test, Y_test, batch_size=_batch_size):
        for i in range(4):
            for j in range(4):
                x = X_batch[(4*i)+j]  # this is a Numpy array with shape (3, 150, 150)
                xt = x.reshape(1,120,160,1)
                p = round(float(model.predict(xt)),3)
                x = x.reshape(120,160)  # this is a Numpy array with shape (1, 3, 150, 150)
                axarr[i,j].imshow(x*255)
                axarr[i,j].axis('off')
                axarr[i,j].set_title(str(p))
        f.suptitle("Sample of Predictions")
        f.show()
        break
        
    # Print sample of predictions along with correct labels 
    for i in range(16):
        print("\nCorrect: \t"+str(Y_test[i]))
        print("Predicted: \t"+str(pred[i]))
        
    time1 = time.time()
    print("time to predict: ", time1-time0)
    _, acc = model.evaluate_generator(test_generator, steps=len(X_test) // _batch_size, verbose=True)
    print('Test Accuracy: %.3f' % (acc * 100))
    return model
    # #return(1/acc)
    
# # results = skopt.forest_minimize(objective, dimensions=SPACE,
# #                                     n_calls=50,
# #                                     base_estimator='RF',
# #                                     acq_func='PI',
# #                                     verbose=True,
# #                                     xi = .25)

if __name__ == "__main__":
    model = main()
