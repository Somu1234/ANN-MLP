import numpy as np

#Creating the dataset. Each Column is one variable.
X = np.array([[1, 0, 1, 0],
              [1, 0, 1, 1],
              [0, 1, 0, 1]])
print ('\nDataset :')
print(X)

#The Class Labels for each row.
y = np.array([[1],
              [1],
              [0]])
print ('\nClass Labels :')
print(y)

#Activation Function.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Derivative for SGD.
def dx_sigmoid(x):
    return x * (1 - x)

#MLP Parameters.
epochs = 1000
lr = 0.1
input_neurons = X.shape[1]
hidden_neurons = 3
output_neurons = 1

#Random Initialisation of weights and biases.
#Hidden Layer weights (3 neurons * 4 variables) and biases (3 neurons).
wh = np.random.uniform(size = (input_neurons, hidden_neurons))
bh = np.random.uniform(size = (1, hidden_neurons))
#Output Layer weights (1 neuron * 3 variables) and biases (1 neuron).
wout = np.random.uniform(size = (hidden_neurons, output_neurons))
bout = np.random.uniform(size = (1, output_neurons))

for i in range(epochs):
    #Feed-Forward.
    hidden_layer1 = np.dot(X, wh)
    hidden_layer = hidden_layer1 + bh
    hidden_output = sigmoid(hidden_layer)
    output_layer1 = np.dot(hidden_output, wout)
    output_layer= output_layer1 + bout
    output = sigmoid(output_layer)
    #Backpropagation of errors thorugh the layers.
    E = y - output
    output_layer_gradient = dx_sigmoid(output)
    hidden_layer_gradient = dx_sigmoid(hidden_output)
    d_output = E * output_layer_gradient
    hidden_layer_error = d_output.dot(wout.T)
    d_hidden = hidden_layer_error * hidden_layer_gradient
    wout += hidden_output.T.dot(d_output) * lr
    bout += np.sum(d_output, axis = 0, keepdims = True) * lr
    wh += X.T.dot(d_hidden) * lr
    bh += np.sum(d_hidden, axis = 0, keepdims = True) * lr
    #Displaying parameters for every 100th Epoch.
    if (i + 1) % 100 == 0:
        print('\nEPOCH : ', i + 1)
        print('Weights Hidden Layer : ')
        print(wh)
        print('Biases Hidden Layer : ')
        print(bh)
        print('Weights Output Layer : ')
        print(wout)
        print('Biases Output Layer : ')
        print(bout)
        print ('Output :')
        print (output)

print ('\nOutput :')
print (output)
