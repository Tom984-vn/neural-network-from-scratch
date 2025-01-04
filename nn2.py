import matplotlib.pyplot as plt
import numpy as np

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

class_num = 3
X, y = create_data(100,class_num)

plt.figure(figsize=(8, 6))
for class_number in range(class_num):  # Loop through each class
    plt.scatter(X[y == class_number, 0], X[y == class_number, 1], label=f'Class {class_number}')
plt.title('Spiral Data')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()