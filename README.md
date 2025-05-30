# Digit Classification using MNIST

This project uses a neural network (built with TensorFlow and Keras) to classify handwritten digits (0–9) from the MNIST dataset.

## Libraries Used
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib

## Overview
This notebook trains a simple neural network on the MNIST dataset, evaluates its performance, and visualizes the results.

jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0

```python
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

```python
x_train.shape
```

    (60000, 28, 28)

```python
y_test
```

    array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

```python
x_train = x_train/255
x_test = x_test/255
```

```python
x_train
```

    array([[[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],

           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],

           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],

           ...,

           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],

           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]],

           [[0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            ...,
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.],
            [0., 0., 0., ..., 0., 0., 0.]]])

```python
x_train_flattern = x_train.reshape(len(x_train), 28 * 28)
x_test_flattern = x_test.reshape(len(x_test), 28 * 28)

x_test_flattern.shape
```

    (10000, 784)

```python
import keras
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid'),
    keras.layers.Dense(10, activation='sigmoid'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flattern, y_train, epochs=10)
```

    Epoch 1/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 3ms/step - accuracy: 0.4533 - loss: 1.8426
    Epoch 2/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 2ms/step - accuracy: 0.8402 - loss: 0.6899
    Epoch 3/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.8805 - loss: 0.4504
    Epoch 4/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9080 - loss: 0.3469
    Epoch 5/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9164 - loss: 0.3026
    Epoch 6/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9228 - loss: 0.2755
    Epoch 7/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.9264 - loss: 0.2598
    Epoch 8/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9321 - loss: 0.2438
    Epoch 9/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9347 - loss: 0.2318
    Epoch 10/10
    1875/1875 ━━━━━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.9392 - loss: 0.2199

    <keras.src.callbacks.history.History at 0x79ff8b038bd0>

```python
model.predict(x_test_flattern)
```

    313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step

    array([[1.3010281e-04, 1.3205688e-04, 2.2731235e-03, ..., 9.8742247e-01,
            5.8414221e-06, 2.0701152e-03],
           [9.7313561e-03, 3.4910447e-03, 9.3154210e-01, ..., 3.5074099e-03,
            7.8299269e-03, 3.1303909e-05],
           [3.9027359e-06, 9.9282169e-01, 4.4270041e-03, ..., 1.7305145e-04,
            1.9281743e-03, 6.1874045e-05],
           ...,
           [1.8177003e-05, 1.7220051e-05, 1.0816568e-03, ..., 2.2461687e-03,
            1.5658563e-03, 1.1031827e-02],
           [5.8637874e-04, 4.6610068e-05, 1.0458647e-03, ..., 4.2683096e-05,
            2.4528123e-02, 1.8455898e-05],
           [5.1031011e-04, 5.7535850e-05, 3.1917179e-03, ..., 2.3680554e-06,
            9.1672358e-05, 1.0167633e-06]], dtype=float32)

```python
y_predict = model.predict(x_test_flattern)
y_predict[0]
```

    313/313 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step

    array([1.3010281e-04, 1.3205688e-04, 2.2731235e-03, 7.8275669e-03,
           1.5119352e-05, 1.2337907e-04, 9.9060692e-08, 9.8742247e-01,
           5.8414221e-06, 2.0701152e-03], dtype=float32)

```python
plt.imshow(x_test[0])
```

    <matplotlib.image.AxesImage at 0x79ff70c7b810>

![](vertopal_164f8fd76e6141728c60ae0ed5ebc4d9/16b08597236cc975abd22da0c5bd046a72011652.png)

```python
np.argmax(y_predict[0])
```

    np.int64(7)

```python
import sklearn as sk
from sklearn.metrics import confusion_matrix
y_predict_labels = [np.argmax(i) for i in y_predict]
cm = sk.metrics.confusion_matrix(y_test, y_predict_labels)
cm


```

    array([[ 954,    0,    3,    1,    1,   15,    4,    2,    0,    0],
           [   0, 1114,    2,    5,    0,    4,    5,    2,    3,    0],
           [  14,    5,  927,   33,   13,    4,   10,   14,   12,    0],
           [   5,    0,   15,  915,    0,   40,    2,   15,   14,    4],
           [   2,    4,    4,    0,  911,    2,   15,    3,    2,   39],
           [  13,    1,    1,   36,    5,  785,   20,    7,   19,    5],
           [   9,    4,    9,    0,    8,   15,  908,    0,    5,    0],
           [   2,   10,   18,    7,    5,    0,    0,  969,    1,   16],
           [   3,    5,    2,   17,    7,   37,   10,    7,  876,   10],
           [   8,    7,    0,    8,   29,   10,    0,   13,    8,  926]])

```python
import seaborn as sns

sns = sns.heatmap(cm, annot=True, fmt='d')
```

![](vertopal_164f8fd76e6141728c60ae0ed5ebc4d9/8f3f105590278cb45083c4da6c06395095ca0ede.png)

## Conclusion
The model achieves high accuracy (>98%) in recognizing handwritten digits using a basic neural network architecture.

## Try it in Colab
[Open In Colab](https://colab.research.google.com/github/rajkachhadiya123/ANN-multiclass-classification/blob/main/ANN_image_project.ipynb)
