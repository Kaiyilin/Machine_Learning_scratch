import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
from ANN import Artificial_Neural_Networks, Losses
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os, sys

def oneHot_encoder(y, num_classes):
    """
    Took me 120ms for ther transformation,
    but only took 4.73 ms for Keras to_categorical
    """
    one_hot_dict = {}
    for i in range(num_classes):
        trans_code = np.zeros((1, num_classes))
        trans_code[0][i] = 1
        one_hot_dict[i] = trans_code
    one_hot_array = []
    for val in y:
        try:
            one_hot_array.append(one_hot_dict[val])
        except KeyError:
            sys.exit(f"The number {val} exceeds the classes you proposed")
    return np.concatenate(one_hot_array, axis=0)

# mnist dataset
num_classes = 10
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()  

train_images_norm = train_images.astype('float32')/255
train_images_forNN = train_images_norm.reshape((60000, 28*28))
train_labels_one_hot = oneHot_encoder(train_labels, num_classes)  

test_images_norm = test_images.astype('float32')/255
test_images_forNN = test_images_norm.reshape((10000, 28*28))
test_labels_one_hot = oneHot_encoder(test_labels, num_classes)  

epochs = 2

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
        print(f"""Epochs_{epoch+1}: {i/600:.1f}% Completed, Loss: {Losses.mse(NN.y_pred, train_labels_one_hot_batch):.2f}""")
        NN.backprop(lr=1e-2)
        #NN.get_weights()
pred = np.argmax(NN.predict(test_images_forNN),axis=1)
conf = confusion_matrix(pred, test_labels)
print(f"model prediction: {pred}")
print(f"model labels: {test_labels}")
plt.plot(loss_list)
plt.savefig('loss.png')
plt.close()
print(conf)
"""
plt.matshow(conf)
plt.xticks([0,1,2,3,4,5,6,7,8,9])
plt.yticks([0,1,2,3,4,5,6,7,8,9])
plt.savefig("confusion_matrix.png")
"""