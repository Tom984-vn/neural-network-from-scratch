import sys
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def create_data(points, classes): #create spiral data
    X = np.zeros((points*classes,2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X,y

X, y = create_data(100,3)

class layer_Dense: #the layer of neurons
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #do not need transpose again (it's the opposite already)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU: #introduce non-linear into model
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax: #scoring for each samples
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #u - max(u) to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
class Loss:  #for many different loss calculation also uses scoring to calculate the actual loss of the model
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #calculate the average loss across all sample
        return data_loss
class Loss_CategoricalCrossentropy(Loss): #Inheritance from actual loss calculation
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7) #prevent infinitive
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood


Dense1 = layer_Dense(2,5)
activation1 = Activation_ReLU()

Dense2 = layer_Dense(5,3)
activation2 = Activation_Softmax()
#actual working
Dense1.forward(X)
activation1.forward(Dense1.output)

Dense2.forward(activation1.output)
activation2.forward(Dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print(loss)