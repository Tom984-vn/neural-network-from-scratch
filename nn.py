import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

X_test, y_test = create_data(100,3)
class Layer_Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

class layer_Dense: #the layer of neurons
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1 = 0, weight_regularizer_L2 = 0, bias_regularizer_L1 = 0, bias_regularizer_L2 = 0):
        self.weights = 0.10 *np.random.randn(n_inputs, n_neurons) #do not need transpose again (it's the opposite already)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_L1
        self.weight_regularizer_l2 = weight_regularizer_L2
        self.bias_regularizer_l1 = bias_regularizer_L1
        self.bias_regularizer_l2 = bias_regularizer_L2
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #the gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims= True)
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0]= -1
            self.dweights += self.weight_regularizer_l1*dL1
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2*self.bias_regularizer_l2 * self.biases
        #gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU: #introduce non-linear into model
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy() #we need to modify the original variable

        self.dinputs[self.inputs <= 0] = 0 #dinputs != inputs
        
class Activation_Softmax_Loss_CategoricalCrossentropy(): #combine both softmax and loss calculate
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    def forward(self, inputs, y_true):  
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1) #to find the index of one-hot coded vectors (argmax is finding the index of the max in the vector)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
class Activation_Softmax: #scoring for each samples
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) #u - max(u) to prevent overflow
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    def backward(self, dvalues):
        #create uninitialize array
        self.inputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:  #for many different loss calculation also uses scoring to calculate the actual loss of the model
    def regularization_loss(self,layer):
        regularization_loss = 0

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights*layer.weights)
        if layer.bias_regularizer_l1 > 0: 
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
        return regularization_loss
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #calculate the average loss across all sample
        return data_loss
class Loss_CategoricalCrossentropy(Loss): #Inheritance from actual loss calculation
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) #prevent infinitive
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
class Optimizer_Adam:
    def __init__(self, learning_rate = 0.001 , decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999 ): #adagrad replace momentum (adagrad is momentum but more complex)
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.interation = 0 
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    def Pre_update_param(self): #learning_rate decaying
        if self.decay: 
            self.current_learning_rate = self.learning_rate * (1. / (1.+ self.decay*self.interation))
    def Update_param(self, layer):
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)          
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1)*layer.dweights #take more from the past momentum than new dweights (beta1 = 0.9)
        layer.bias_momentums = self.beta_1* layer.bias_momentums + (1 - self.beta_1)*layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.interation + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.interation + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2)* layer.dweights**2  #take more from the past than current dweights
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1- self.beta_2) * layer.dbiases**2
        
        weight_cache_corrected = layer.weight_cache / (1- self.beta_2 ** (self.interation + 1)) #corrected because at first momentum and cache were very small
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2**(self.interation + 1))  
    
        layer.weights += -self.current_learning_rate* weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate* bias_momentums_corrected / (np.sqrt(bias_cache_corrected)+ self.epsilon)
    def Post_update_param(self):
        self.interation += 1
#Initialization
Dense1 = layer_Dense(2,512, weight_regularizer_L2= 5e-4, bias_regularizer_L2= 5e-4)   
activation1 = Activation_ReLU()
Drop_out_1 = Layer_Dropout(0.1)
Dense2 = layer_Dense(512,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_Adam(learning_rate = 0.05, decay=5e-5)
for epochs in range(10001):
    #actual working
    Dense1.forward(X)
    activation1.forward(Dense1.output)
    Drop_out_1.forward(activation1.output)

    Dense2.forward(Drop_out_1.output)

    data_loss = loss_activation.forward(Dense2.output, y)

    regularization_loss = loss_activation.loss.regularization_loss(Dense1) + loss_activation.loss.regularization_loss(Dense2)
    loss = data_loss + regularization_loss
    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    

    if not epochs % 100:
        print(f'epoch {epochs}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}, ' +
        f'data_loss: {data_loss: .3f}' + 
        f'reg_loss: {regularization_loss: .3f}' +
        f'lr: {optimizer.current_learning_rate}')
    loss_activation.backward(loss_activation.output, y_test)
    Dense2.backward(loss_activation.dinputs)
    Drop_out_1.backward(Dense2.dinputs)
    activation1.backward(Drop_out_1.dinputs)
    Dense1.backward(activation1.dinputs)

    optimizer.Pre_update_param()
    optimizer.Update_param(Dense1)
    optimizer.Update_param(Dense2)
    optimizer.Post_update_param()

#testing
Dense1.forward(X_test)
activation1.forward(Dense1.output)

Dense2.forward(activation1.output)

loss = loss_activation.forward(Dense2.output, y)

predictions = np.argmax(loss_activation.output, axis = 1)
if len(y_test.shape) == 2:
    y = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'validation, acc: {accuracy:.3f}, lost: {loss:.3f}')

