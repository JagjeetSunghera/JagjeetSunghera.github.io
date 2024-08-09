## **Title:** 
Introduction to CNN using Keras and Tensorflow.

## **Objective:** 
Creating an image classification model in Keras framework to identify classes in a toy dataset.

### **Difficulty Level:** 
Easy


## **Introduction:**
Image classification model have been in usage for decades the first classification models used large number of perceptrons and were useful with low quality image. however the these where not scalable for large images and images with broad color spectrum. Putting it in easy words the a 64*64 image took almost 64*64 neuron on the first layer and then it reduces. there was also issues with locality and finding pattern in image if the image is moved a bit in any direction.


Thus utilization of CNN took advantages turn where kernels are used imagine this as matrices which can be multiplied with small patches of images and activation function like rectified linear unit (relu) are used to make the result into 0 and 1 this make sure that during a optimization the gradient from the loss function is not very small which lead to longer duration for optimization. A CNN block have multiple kernels these allow us to find mutiple textures and patterns in the image.


lets get started to make it more possible.

### Loading libraries:

we need the following libraries/packages to be imported into the project for makeing tensors, creating datagenerator, and ploting images.


```python
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
data=ImageDataGenerator()
train_path='/content/drive/MyDrive/PROJECTS/constems/toy_train'
test_path='/content/drive/MyDrive/PROJECTS/constems/toy_val'
```


```python
x_train,y_train=next(data.flow_from_directory(train_path,target_size=(64,64),batch_size=-1))
```

    Found 1000 images belonging to 2 classes.
    


```python
x_test,y_test=next(data.flow_from_directory(test_path,target_size=(64,64),batch_size=-1))
```

    Found 200 images belonging to 2 classes.
    


```python
#Cheaking the length of the data
len(y_train),len(y_test)
```




    (999, 199)




```python
#Cheaking Train Data
y_train[22]
```




    array([1., 0.], dtype=float32)




```python
plt.imshow(x_train[22].astype('uint8'))
```




    <matplotlib.image.AxesImage at 0x7bb0941b9cc0>




    
![png]({{ '/_posts/constemintern/output_10_1.png' | relative_url }})
    



```python
y_train[1]
```




    array([1., 0.], dtype=float32)




```python
plt.imshow(x_train[1].astype('uint8'))
```




    <matplotlib.image.AxesImage at 0x7bb0940502e0>




    
![png]({{ '/_posts/constemintern/output_12_1.png' | relative_url }})
    



```python
#Cheaking Test data
y_test[5]
```




    array([1., 0.], dtype=float32)




```python
plt.imshow(x_test[5].astype('uint8'))
```




    <matplotlib.image.AxesImage at 0x7bb0940c6c80>




    
![png]({{ '/_posts/constemintern/output_14_1.png' | relative_url }})
    



```python
#Coverting data type to float32
```


### As now the data is generated now we can procced to model buliding stage


# Importing Required Library


```python

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer,MaxPool2D,Dense,Dropout,Conv2D,Flatten
```

we are using only 3 kernels and kernel size of 4*4. the reson for using only 6 kerenel feature is simple we know inorder to identify a box from circle we only need to identify the edges and corners. this make it easy.


```python
#initiating Sequential model object and adding layers

model= Sequential()
model.add(InputLayer(input_shape=(64,64,3),dtype='float64'))
model.add(Conv2D(6,kernel_size=(4,4),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(4,kernel_size=(4,4),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=3,activation='relu'))
model.add(Dense(units=2,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2,activation='softmax'))

```


```python
#Compiling the model with adam optimizer and categorical CrossEntropy
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

```


```python
model.summary()

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_1"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">61</span>, <span style="color: #00af00; text-decoration-color: #00af00">61</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">294</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">6</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">27</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │             <span style="color: #00af00; text-decoration-color: #00af00">388</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">13</span>, <span style="color: #00af00; text-decoration-color: #00af00">4</span>)           │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">676</span>)                 │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │           <span style="color: #00af00; text-decoration-color: #00af00">2,031</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_4 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                   │               <span style="color: #00af00; text-decoration-color: #00af00">8</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                   │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">2</span>)                   │               <span style="color: #00af00; text-decoration-color: #00af00">6</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,727</span> (10.65 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">2,727</span> (10.65 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>




```python
### Run the program for 100, 200, and 300 iterations, respectively. Follow this by a report on the final accuracy and loss on the evaluation data.

```


```python
# with 100 Epochs or Iterations
history100=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100,verbose=True)

```

    Epoch 1/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m7s[0m 95ms/step - accuracy: 0.5382 - loss: 0.8564 - val_accuracy: 0.7839 - val_loss: 0.5615
    Epoch 2/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m5s[0m 6ms/step - accuracy: 0.7573 - loss: 0.5473 - val_accuracy: 0.8442 - val_loss: 0.4815
    Epoch 3/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8449 - loss: 0.4663 - val_accuracy: 0.9196 - val_loss: 0.4103
    Epoch 4/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.8899 - loss: 0.4172 - val_accuracy: 0.9397 - val_loss: 0.3689
    Epoch 5/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.8972 - loss: 0.4014 - val_accuracy: 0.9548 - val_loss: 0.3411
    Epoch 6/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 9ms/step - accuracy: 0.9130 - loss: 0.3657 - val_accuracy: 0.9548 - val_loss: 0.3237
    Epoch 7/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 6ms/step - accuracy: 0.9181 - loss: 0.3496 - val_accuracy: 0.9548 - val_loss: 0.3066
    Epoch 8/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9280 - loss: 0.3377 - val_accuracy: 0.9648 - val_loss: 0.2887
    Epoch 9/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9503 - loss: 0.2988 - val_accuracy: 0.9698 - val_loss: 0.2743
    Epoch 10/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9436 - loss: 0.2876 - val_accuracy: 0.9698 - val_loss: 0.2608
    Epoch 11/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9312 - loss: 0.2948 - val_accuracy: 0.9749 - val_loss: 0.2476
    Epoch 12/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9422 - loss: 0.2777 - val_accuracy: 0.9849 - val_loss: 0.2338
    Epoch 13/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9358 - loss: 0.2717 - val_accuracy: 0.9849 - val_loss: 0.2199
    Epoch 14/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9438 - loss: 0.2529 - val_accuracy: 0.9849 - val_loss: 0.2096
    Epoch 15/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9433 - loss: 0.2589 - val_accuracy: 0.9849 - val_loss: 0.2012
    Epoch 16/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9530 - loss: 0.2327 - val_accuracy: 0.9899 - val_loss: 0.1917
    Epoch 17/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9227 - loss: 0.2648 - val_accuracy: 0.9698 - val_loss: 0.2144
    Epoch 18/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9340 - loss: 0.2553 - val_accuracy: 0.9899 - val_loss: 0.1767
    Epoch 19/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9358 - loss: 0.2434 - val_accuracy: 0.9899 - val_loss: 0.1710
    Epoch 20/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9474 - loss: 0.2169 - val_accuracy: 0.9899 - val_loss: 0.1645
    Epoch 21/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9555 - loss: 0.2049 - val_accuracy: 0.9899 - val_loss: 0.1596
    Epoch 22/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9505 - loss: 0.2188 - val_accuracy: 0.9899 - val_loss: 0.1537
    Epoch 23/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9492 - loss: 0.2111 - val_accuracy: 0.9899 - val_loss: 0.1479
    Epoch 24/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9420 - loss: 0.2218 - val_accuracy: 0.9899 - val_loss: 0.1444
    Epoch 25/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9426 - loss: 0.2123 - val_accuracy: 0.9899 - val_loss: 0.1395
    Epoch 26/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9511 - loss: 0.1978 - val_accuracy: 0.9899 - val_loss: 0.1352
    Epoch 27/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9656 - loss: 0.1787 - val_accuracy: 0.9899 - val_loss: 0.1320
    Epoch 28/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9424 - loss: 0.2019 - val_accuracy: 0.9899 - val_loss: 0.1273
    Epoch 29/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9519 - loss: 0.1922 - val_accuracy: 0.9899 - val_loss: 0.1243
    Epoch 30/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9493 - loss: 0.1927 - val_accuracy: 0.9950 - val_loss: 0.1208
    Epoch 31/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9302 - loss: 0.2280 - val_accuracy: 0.9950 - val_loss: 0.1184
    Epoch 32/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9443 - loss: 0.2001 - val_accuracy: 0.9950 - val_loss: 0.1153
    Epoch 33/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9537 - loss: 0.1816 - val_accuracy: 0.9950 - val_loss: 0.1124
    Epoch 34/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9597 - loss: 0.1683 - val_accuracy: 0.9950 - val_loss: 0.1091
    Epoch 35/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9273 - loss: 0.2228 - val_accuracy: 0.9899 - val_loss: 0.1087
    Epoch 36/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9564 - loss: 0.1732 - val_accuracy: 0.9899 - val_loss: 0.1033
    Epoch 37/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9492 - loss: 0.1852 - val_accuracy: 0.9950 - val_loss: 0.0989
    Epoch 38/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9598 - loss: 0.1656 - val_accuracy: 0.9950 - val_loss: 0.0963
    Epoch 39/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9320 - loss: 0.2103 - val_accuracy: 0.9950 - val_loss: 0.0948
    Epoch 40/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9449 - loss: 0.1912 - val_accuracy: 0.9950 - val_loss: 0.0920
    Epoch 41/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9549 - loss: 0.1686 - val_accuracy: 0.9950 - val_loss: 0.0983
    Epoch 42/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9546 - loss: 0.1693 - val_accuracy: 0.9950 - val_loss: 0.0947
    Epoch 43/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9437 - loss: 0.1892 - val_accuracy: 0.9950 - val_loss: 0.0920
    Epoch 44/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9414 - loss: 0.1963 - val_accuracy: 0.9950 - val_loss: 0.0938
    Epoch 45/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9556 - loss: 0.1653 - val_accuracy: 0.9950 - val_loss: 0.0900
    Epoch 46/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9493 - loss: 0.1776 - val_accuracy: 0.9950 - val_loss: 0.0869
    Epoch 47/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9456 - loss: 0.1871 - val_accuracy: 0.9950 - val_loss: 0.0864
    Epoch 48/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9495 - loss: 0.1733 - val_accuracy: 0.9950 - val_loss: 0.0847
    Epoch 49/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9506 - loss: 0.1745 - val_accuracy: 0.9950 - val_loss: 0.0849
    Epoch 50/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9400 - loss: 0.1951 - val_accuracy: 0.9950 - val_loss: 0.0844
    Epoch 51/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9496 - loss: 0.1747 - val_accuracy: 0.9950 - val_loss: 0.0817
    Epoch 52/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9518 - loss: 0.1679 - val_accuracy: 0.9950 - val_loss: 0.0798
    Epoch 53/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9469 - loss: 0.1766 - val_accuracy: 0.9950 - val_loss: 0.0784
    Epoch 54/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9456 - loss: 0.1816 - val_accuracy: 0.9950 - val_loss: 0.0780
    Epoch 55/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9527 - loss: 0.1643 - val_accuracy: 0.9950 - val_loss: 0.0765
    Epoch 56/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9694 - loss: 0.1272 - val_accuracy: 0.9950 - val_loss: 0.0761
    Epoch 57/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9530 - loss: 0.1660 - val_accuracy: 0.9950 - val_loss: 0.0750
    Epoch 58/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9395 - loss: 0.1918 - val_accuracy: 0.9950 - val_loss: 0.0741
    Epoch 59/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9393 - loss: 0.1932 - val_accuracy: 0.9950 - val_loss: 0.0745
    Epoch 60/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9412 - loss: 0.1877 - val_accuracy: 0.9950 - val_loss: 0.0735
    Epoch 61/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9428 - loss: 0.1855 - val_accuracy: 0.9950 - val_loss: 0.0722
    Epoch 62/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9467 - loss: 0.1771 - val_accuracy: 0.9950 - val_loss: 0.0719
    Epoch 63/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9321 - loss: 0.2098 - val_accuracy: 0.9950 - val_loss: 0.0715
    Epoch 64/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 7ms/step - accuracy: 0.9471 - loss: 0.1725 - val_accuracy: 0.9950 - val_loss: 0.0700
    Epoch 65/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9612 - loss: 0.1466 - val_accuracy: 0.9950 - val_loss: 0.0706
    Epoch 66/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 8ms/step - accuracy: 0.9507 - loss: 0.1703 - val_accuracy: 0.9950 - val_loss: 0.0681
    Epoch 67/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9624 - loss: 0.1434 - val_accuracy: 0.9950 - val_loss: 0.0658
    Epoch 68/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9462 - loss: 0.1778 - val_accuracy: 0.9950 - val_loss: 0.0647
    Epoch 69/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9583 - loss: 0.1511 - val_accuracy: 0.9950 - val_loss: 0.0642
    Epoch 70/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9523 - loss: 0.1636 - val_accuracy: 0.9950 - val_loss: 0.0635
    Epoch 71/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9556 - loss: 0.1578 - val_accuracy: 0.9950 - val_loss: 0.0635
    Epoch 72/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9670 - loss: 0.1301 - val_accuracy: 0.9950 - val_loss: 0.0624
    Epoch 73/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9326 - loss: 0.2052 - val_accuracy: 0.9950 - val_loss: 0.0621
    Epoch 74/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9295 - loss: 0.2127 - val_accuracy: 0.9950 - val_loss: 0.0620
    Epoch 75/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9599 - loss: 0.1469 - val_accuracy: 0.9950 - val_loss: 0.0620
    Epoch 76/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9257 - loss: 0.2259 - val_accuracy: 0.9950 - val_loss: 0.0619
    Epoch 77/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9381 - loss: 0.1938 - val_accuracy: 0.9950 - val_loss: 0.0616
    Epoch 78/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9581 - loss: 0.1507 - val_accuracy: 0.9950 - val_loss: 0.0608
    Epoch 79/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9576 - loss: 0.1499 - val_accuracy: 0.9950 - val_loss: 0.0602
    Epoch 80/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9602 - loss: 0.1469 - val_accuracy: 0.9950 - val_loss: 0.0598
    Epoch 81/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9359 - loss: 0.1998 - val_accuracy: 0.9950 - val_loss: 0.0595
    Epoch 82/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9536 - loss: 0.1618 - val_accuracy: 0.9950 - val_loss: 0.0589
    Epoch 83/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9469 - loss: 0.1762 - val_accuracy: 0.9950 - val_loss: 0.0586
    Epoch 84/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9291 - loss: 0.2153 - val_accuracy: 0.9950 - val_loss: 0.0583
    Epoch 85/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9540 - loss: 0.1605 - val_accuracy: 0.9950 - val_loss: 0.0586
    Epoch 86/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 4ms/step - accuracy: 0.9458 - loss: 0.1771 - val_accuracy: 0.9950 - val_loss: 0.0586
    Epoch 87/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9370 - loss: 0.1990 - val_accuracy: 0.9950 - val_loss: 0.0583
    Epoch 88/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9463 - loss: 0.1784 - val_accuracy: 0.9950 - val_loss: 0.0581
    Epoch 89/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9588 - loss: 0.1488 - val_accuracy: 0.9950 - val_loss: 0.0574
    Epoch 90/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9419 - loss: 0.1862 - val_accuracy: 0.9950 - val_loss: 0.0572
    Epoch 91/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 6ms/step - accuracy: 0.9417 - loss: 0.1901 - val_accuracy: 0.9950 - val_loss: 0.0577
    Epoch 92/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9590 - loss: 0.1469 - val_accuracy: 0.9950 - val_loss: 0.0566
    Epoch 93/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9512 - loss: 0.1645 - val_accuracy: 0.9950 - val_loss: 0.0562
    Epoch 94/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9328 - loss: 0.2080 - val_accuracy: 0.9950 - val_loss: 0.0565
    Epoch 95/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9541 - loss: 0.1602 - val_accuracy: 1.0000 - val_loss: 0.0561
    Epoch 96/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9349 - loss: 0.1992 - val_accuracy: 1.0000 - val_loss: 0.0559
    Epoch 97/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9402 - loss: 0.1915 - val_accuracy: 1.0000 - val_loss: 0.0558
    Epoch 98/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9449 - loss: 0.1804 - val_accuracy: 1.0000 - val_loss: 0.0558
    Epoch 99/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9416 - loss: 0.1874 - val_accuracy: 0.9950 - val_loss: 0.0568
    Epoch 100/100
    [1m32/32[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 5ms/step - accuracy: 0.9439 - loss: 0.1841 - val_accuracy: 0.9950 - val_loss: 0.0569
    


```python
plt.plot(history100.history['loss'],label='Train')
plt.plot(history100.history['val_loss'],label='Test')
plt.legend()
plt.show()

```


    
![png](https://jagjeetsunghera.github.io/_posts/constemintern/output_25_0.png)
    


The above graph shows that low bias and low variance. Also the error line of validation is below the training line which signs toward the position that we have good genralization. this can be also confirmed by the graph below which shows that the data is properly learned by the model and is now able to classify boxes from circles.


```python
plt.plot(history100.history['accuracy'],label='Train')
plt.plot(history100.history['val_accuracy'],label='Test')
plt.legend()
plt.show()

```


    
![png](https://jagjeetsunghera.github.io/_posts/constemintern/output_27_0.png)
    



```python

```
