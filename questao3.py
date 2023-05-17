import numpy as np


def activate_function(value):
    if value > 0:
        return 1
    else:
        return 0


def perceptron(value, weight, bias):
    z = np.dot(value, weight) + bias
    return activate_function(z)


def train_perceptron(data_x_train: np.array, data_y_train: np.array, percent_learning_rate, trainning_sessions):

    weight = np.random.rand(2)
    bias = np.random.rand(1)

    for count in range(trainning_sessions):
        for input_data, output_expected in zip(data_x_train, data_y_train):
            output_prediction = perceptron(input_data, weight, bias)

            weight = weight + percent_learning_rate * (output_expected - output_prediction) * input_data
            bias = bias + percent_learning_rate * (output_expected - output_prediction)

    return weight, bias


x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([1, 0, 0, 0])

w, b = train_perceptron(data_x_train=x_train, data_y_train=y_train, percent_learning_rate=0.1, trainning_sessions=100)


x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([1, 0, 0, 0])

for x_i, y_true in zip(x_test, y_test):
    y_pred = perceptron(x_i, w, b)
    print(f'Entrada: {x_i}.\nSaída Esperada: {y_true} | Saída do Modelo: {y_pred}\n')
