import numpy as np
from numpy import linalg as LA

w = [np.zeros(784) for _ in range(10)]
bias = [0] * 10


def find_group(data):
    maximum = np.dot(data, w[0]) + bias[0]
    max_index = 0
    for i in range(1, 10):
        dot_product = np.dot(data, w[i]) + bias[i]
        if dot_product > maximum:
            maximum = dot_product
            max_index = i
    return max_index


def handle_error(data, right_group, wrong_group, is_mira):
    if is_mira:
        norm = LA.norm(data)
        t = (np.dot((w[wrong_group] - w[right_group]), data) + bias[wrong_group] - bias[right_group] + 1) / (
                2 * norm * norm + 2)
    else:
        t = 1
    w[right_group] += t * data
    bias[right_group] += t * 1
    w[wrong_group] -= t * data
    bias[wrong_group] -= t * 1


def perceptron_train(X_train, y_train, is_mira):
    if is_mira:
        rounds = 5
    else:
        rounds = 6
    for _ in range(rounds):
        for i in range(len(X_train)):
            data = X_train[i]
            estimated_group = find_group(data)
            right_group = y_train[i]
            if estimated_group == right_group:
                continue
            else:
                handle_error(data, right_group, estimated_group, is_mira)
    f = open('./perceptron.txt', 'w')
    for i in range(len(w)):
        print(w[i], bias[i])

    f.close()


def perceptron_test(X_test, y_test):
    rights, wrongs = 0, 0
    for i in range(len(X_test)):
        data = X_test[i]
        estimated_group = find_group(data)
        if estimated_group == y_test[i]:
            rights += 1
        else:
            wrongs += 1
    size = len(X_test)
    accuracy = (100 * rights) / size
    return accuracy


