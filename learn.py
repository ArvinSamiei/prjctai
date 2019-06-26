import numpy as np
import os
import perceptron_mira

is_mira = True


def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


images, labels = load_mnist('./data')

X_train, y_train = load_mnist('./data', kind='train')
X_test, y_test = load_mnist('./data', kind='t10k')

X_train = X_train.copy()
X_test = X_test.copy()

X_train.setflags(write=1)
X_test.setflags(write=1)

X_train[X_train > 0] = 1
X_test[X_test > 0] = 1


perceptron_mira.perceptron_train(X_train, y_train, is_mira)
print(perceptron_mira.perceptron_test(X_test, y_test))