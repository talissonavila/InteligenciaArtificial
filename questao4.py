import numpy as np


def identity(value):
    return value


class Adaline:
    def __init__(self, input_size, percent_learning_rate=0.01, trainning_sessions=100):
        self.input_size = input_size
        self.percent_learning_rate = percent_learning_rate
        self.trainning_sessions = trainning_sessions
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)

    def forward(self, value):
        z = np.dot(value, self.weights) + self.bias
        return z

    def backward(self, value, output_expected, output_prediction):
        error = output_expected - output_prediction
        d_weights = -2 * self.percent_learning_rate * error * value
        d_bias = -2 * self.percent_learning_rate * error
        return d_weights, d_bias

    def train(self, data_x_train, data_y_train):
        for epoch in range(self.trainning_sessions):
            for input_data, output_expected in zip(data_x_train, data_y_train):
                output_prediction = self.forward(input_data)
                d_weights, d_bias = self.backward(input_data, output_expected, output_prediction)
                self.weights = self.weights - d_weights
                self.bias = self.bias - d_bias

    def predict(self, value):
        output_prediction = self.forward(value)
        output_prediction = identity(output_prediction)
        return output_prediction


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([1, 0, 0, 0])

model = Adaline(input_size=2, percent_learning_rate=0.1, trainning_sessions=100)
model.train(x_train, y_train)

x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([1, 0, 0, 0])

for x_i, y_true in zip(x_test, y_test):
    y_pred = model.predict(x_i)
    print(f'Entrada: {x_i}.\nSaída Esperada: {y_true} | Saída do Modelo: {y_pred}\n')
