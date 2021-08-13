import numpy as np 

class Activation(object):
    @staticmethod
    def relu(x):
        return x if x > 0 else 0

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        return np.exp(x)/sum(np.exp(x))


class Activation_deriv(object):
    @staticmethod
    def relu_derive(x):
        return 1 if x >0 else 0
    
    @staticmethod
    def sigmoid_derive(x):
        return Activation.sigmoid(x)*(1-Activation.sigmoid(x))

    @staticmethod
    def softmax_derive(x):
        return  Activation.softmax(x)*(1-Activation.softmax(x))

class Losses(object):
    @staticmethod
    def mse(real, pred):
        """
        mean squared error 
        the equation refered as 
        ((real - pred) ** 2)/num_of_array
        """
        #cost = (1/(m)) * (loss' * loss)
        loss = np.square(pred - real).sum() / len(pred)
        #print(loss)

        return loss
        

class Artificial_Neural_Networks(object):

    def __init__(self, input_size: int, num_hid_neurones: int, num_class: int):
        """
        init the number of neurones for input, hidden, and output layers
        the number must equal to integer
        """
        # The weights and biases should be the huge array or list architecture 
        # that contains the weights and bias  

        self.input_size = input_size
        self.weights1   = np.random.randn(input_size, num_hid_neurones)
        self.bias1 = np.random.randn(num_hid_neurones)
        self.weights2   = np.random.rand(num_hid_neurones, num_class)
        self.bias2 = np.random.random(num_class)
        self.y          = num_class
        #self.y_pred     = np.zeros(self.y.shape)
    
    def feedforward(self, X, Y):
        """
        activation function is defined in another class
        batch_won't affect the result of feed forward
        """
        self.input = X
        self.z1 = np.dot(self.input, self.weights1) 
        self.layer1 = Activation.sigmoid(self.z1 + self.bias1)

        self.z2 = np.dot(self.layer1, self.weights2) 
        self.y_pred = Activation.sigmoid(self.z2 + self.bias2)

        self.y = Y
        self.loss = Losses.mse(self.y, self.y_pred)

    def backprop(self, lr = 1e-4):
        """
        docstring
        """
        self.grad_y_pred = 2.0 * (self.y_pred - self.y) *Activation_deriv.sigmoid_derive(self.y_pred)
        self.grad_weights2 = np.dot(self.layer1.T, self.grad_y_pred)
        self.grad_layer1 = np.dot(self.grad_y_pred, self.weights2.T)
        self.grad_weights1 = np.dot(self.input.T, self.grad_layer1 * Activation_deriv.sigmoid_derive(self.layer1))

        self.weights1 -= lr*self.grad_weights1
        self.bias1 -= lr*np.mean(self.grad_layer1) 
        self.weights2 -= lr*self.grad_weights2
        self.bias2 -= lr*np.mean(self.grad_y_pred)

    def get_weights(self):
        print(self.weights1)
        print(self.weights2)
        print(self.bias1)
        print(self.bias2)
        print("--"*30)


    def fit(self, iterations, batch_size, ):
        loss_list = []
        for iter in range(iterations):
            Artificial_Neural_Networks.feedforward
            Artificial_Neural_Networks.backprop

        return loss_list
    
    def predict(self, testX):
        z = Activation.sigmoid(np.dot(testX, self.weights1) + self.bias1)
        z_pred = Activation.sigmoid(np.dot(z, self.weights2) + self.bias2)
        return z_pred