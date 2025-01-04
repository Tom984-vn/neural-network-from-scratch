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

class layer_Dense: #the layer of neurons
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) #do not need transpose again (it's the opposite already)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        #the gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims= True)
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
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples
class Optimizer_SGD:
    def __init__(self, learning_rate = 1., decay = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.interation = 0
    def Pre_update_param(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1.+ self.decay*self.interation))
    def Update_param(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    def Post_update_param(self):
        self.interation += 1
# For plotting accuracy and loss
accuracy_history = []
loss_history = []
#Initialization
Dense1 = layer_Dense(2,64)       
activation1 = Activation_ReLU()
Dense2 = layer_Dense(64,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD()
for epochs in range(10001):
    #actual working
    Dense1.forward(X)
    activation1.forward(Dense1.output)

    Dense2.forward(activation1.output)

    loss = loss_activation.forward(Dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis = 1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)
    
    # Log accuracy and loss every 100 epochs
    if not epochs % 100:
        accuracy_history.append(accuracy)
        loss_history.append(loss)
        print(f'epoch {epochs}, acc: {accuracy:.3f}, loss: {loss:.3f}')

    if not epochs % 100:
        print(f'epoch {epochs}, ' +
        f'acc: {accuracy:.3f}, ' +
        f'loss: {loss:.3f}')
    loss_activation.backward(loss_activation.output, y)
    Dense2.backward(loss_activation.dinputs)
    activation1.backward(Dense2.dinputs)
    Dense1.backward(activation1.dinputs)

    optimizer.Update_param(Dense1)
    optimizer.Update_param(Dense2)
# Animation setup
fig, ax = plt.subplots(2, 1, figsize=(8, 8))
# Accuracy plot
ax[0].set_xlim(0, 10000)
ax[0].set_ylim(0, 1)
ax[0].set_title("Training Accuracy")
line_acc, = ax[0].plot([], [], color='blue', label="Accuracy")
ax[0].legend()
ax[0].grid()
# Loss plot
ax[1].set_xlim(0, 10000)
ax[1].set_ylim(0, max(loss_history))
ax[1].set_title("Training Loss")
line_loss, = ax[1].plot([], [], color='red', label="Loss")
ax[1].legend()
ax[1].grid()

def update(frame):
    line_acc.set_data(range(0, frame * 100, 100), accuracy_history[:frame])
    line_loss.set_data(range(0, frame * 100, 100), loss_history[:frame])
    return line_acc, line_loss

ani = FuncAnimation(fig, update, frames=len(accuracy_history), blit=True, interval=100)

plt.tight_layout()
plt.show()


