#	Assignment 3
#	BIPOLAR OR, NAND ,NOR and AND from adaline learning using McCulloch-Pitts neural network 

import numpy as np 
import matplotlib.pyplot as plt 

def gate():
    print("Select gate:\n1.OR \n2.NAND\n3.NOR\n4.AND")
    choice=input("Choose:")
    choice = int(choice)
    if choice == 1:
        features = np.array(
            [   
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]
            ])
        labels = np.array([-1, 1, 1, 1])
    elif choice == 2:
        features = np.array(
            [   
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]
            ])
        labels = np.array([1, 1, 1, -1])
    elif choice == 3:
        features = np.array(
            [   
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]
            ])
        labels = np.array([1, -1, -1, -1])
    elif choice == 4:
        features = np.array(
            [   
                [-1, -1],
                [-1, 1],
                [1, -1],
                [1, 1]
            ])
        labels = np.array([-1, -1, -1, 1])
    else:
        print("Invalid choice, select 1 ,2 ,3 or 4")

    return (features,labels)

# create NeuralNetwork class
class NeuralNetwork:

    # intialize variables in class
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        self.weights = np.array([[.50], [.50]])
        self.bias = 0.1
        self.learning_rate = 0.2
        self.error_history = []
        self.epoch_list = []

    # train the neural net for 100 iterations
    def train(self, epochs=100):
        for epoch in range(epochs):
            sum_squared_error = 0.0
            for j in range(self.inputs.shape[0]):    
                actual = self.outputs[j]
                x1 = self.inputs[j][0]
                x2 = self.inputs[j][1]
                unit = (x1 * self.weights[0]) + (x2 * self.weights[1]) + self.bias
                error = actual - unit
                # print("error =", error)
                sum_squared_error += error * error
                self.weights[0] += self.learning_rate * error * x1
                self.weights[1] += self.learning_rate * error * x2
                self.bias += self.learning_rate * error
            # keep track of the sum_squar_error history over each epoch
            self.error_history.append(np.average(np.abs(sum_squared_error)))
            self.epoch_list.append(epoch)

    # function to predict output on new and unseen input data                               
    def predict(self, new_input):
        prediction = (new_input[0][0] * self.weights[0]) + (new_input[0][1] * self.weights[1]) + self.bias
        # prediction = np.round(prediction,0)
        # print(new_input,prediction)
        return prediction

#choose logical gate
inputs, outputs = gate()
print("Input to neural network:\n",inputs,outputs,"\n")

# create neural network   
NN = NeuralNetwork(inputs, outputs)
# train neural network
NN.train()
# create two new examples to predict                                   
example = np.array([[1, 1]])
example_1 = np.array([[1, -1]])
example_2 = np.array([[-1, 1]])
example_3 = np.array([[-1, -1]])

# print the predictions for both examples                                   
print("Predictions:")
print('Input: ', example_3[0], 'Output: ',NN.predict(example_3))
print('Input: ', example_2[0], 'Output: ',NN.predict(example_2))
print('Input: ', example_1[0], 'Output: ',NN.predict(example_1))
print('Input: ', example[0], 'Output: ',NN.predict(example))

# plot the error over the entire training duration
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Sum Squared Error')
plt.show()