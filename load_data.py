import numpy as np
import random as rd

x = np.load('./X.npy')
y = np.load('./Y.npy')


def load_signlang(size=2062):
    x_data = list()
    y_data = list()
    size = size
    for image, lable in zip(x, y):
        if size is 1:
            rd.shuffle(x_data)
            rd.shuffle(y_data)
            return (x_data, y_data)
        x_data.append(image)
        y_data.append(lable)
        size -= 1
    return x_data, y_data

if __name__ == '__main__':
    a = load_signlang()
    print(a[1][1])