from os import pread
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical  
from ANN import Artificial_Neural_Networks, Losses
import matplotlib.pyplot as plt
# mnist dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()  

train_images_norm = train_images.astype('float32')/255
train_images_forNN = train_images_norm.reshape((60000, 28*28))
train_labels_one_hot = to_categorical(train_labels)  

test_images_norm = test_images.astype('float32')/255
test_images_forNN = test_images_norm.reshape((10000, 28*28))
test_labels_one_hot = to_categorical(test_labels)  

epochs = 3

NN = Artificial_Neural_Networks(28*28, 512, 10)

batch_size = 32
batch_num = (train_images_forNN.shape[0]) // batch_size +1

loss_list = []
for epoch in range(epochs):
    for i in range(0, 60000, 32):
        train_images_batch = train_images_forNN[i:i+batch_size]
        train_labels_one_hot_batch = train_labels_one_hot[i:i+batch_size]
        NN.feedforward(train_images_batch, train_labels_one_hot_batch)
        loss_list.append(NN.loss)
        print(f"Loss: {Losses.mse(NN.y_pred, train_labels_one_hot_batch)}")
        NN.backprop(lr=1e-2)
        #NN.get_weights()
pred = np.argmax(NN.predict(test_images_forNN),axis=1)
print(f"model prediction: {pred}")
print(f"model labels: {test_labels}")
plt.plot(loss_list)
plt.savefig('loss.png')