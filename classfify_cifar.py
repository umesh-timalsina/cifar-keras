import tensorflow as tf
import numpy as np
from utils import print_matrix_detail, plot_image


# Step One:  Load and Visualize and Data
cifar_dataset = tf.keras.datasets.cifar10
(CIFAR_X_train, CIFAR_y_train), (CIFAR_X_test, CIFAR_y_test) = cifar_dataset.load_data()
print_matrix_detail(CIFAR_X_train, "CIFAR_X_train")
print_matrix_detail(CIFAR_y_test, "CIFAR_y_train")
print_matrix_detail(CIFAR_X_test, "CIFAR_X_test")
print_matrix_detail(CIFAR_y_test, "CIFAR_y_test")
# plot_image(CIFAR_X_test[0:25], CIFAR_y_test[0:25])

# Normalize and flatten the input images
CIFAR_X_train = CIFAR_X_train.reshape(50000, 3072)/255
CIFAR_X_test = CIFAR_X_test.reshape(10000, 3072)/255


def model(input_dim, num_classes, num_hidden=2, num_hidden_units=256):
    """
    Build and return the model
    Args:
        input_shape: A tuple for input shape
        num_classes: Number of classes in the softmax
        num_hidden: Number of hidden layers 
        num_hidden_units: number of of hidden units
    Returns:
        the keras model
    """
    input_layer = tf.keras.layers.Dense(num_hidden_units,
                                        activation=tf.nn.relu,
                                        input_dim=input_dim)
    hidden_layers = []
    for i in range(num_hidden-1):
        hidden_layers.append(tf.keras.layers.Dense(num_hidden_units,
                                                   activation=tf.nn.relu))
    output_layer = tf.keras.layers.Dense(num_classes,
                                         activation=tf.nn.softmax)
    model = tf.keras.Sequential([input_layer, hidden_layers[0], output_layer])
    return model

model_cifar = model(input_dim=3072,
                    num_classes=10,
                    num_hidden=2)
# model_cifar.compile()
model_cifar.summary()

# Step Three: Compile and Train the model
model_cifar.compile(optimizer=tf.keras.optimizers.SGD(lr=0.01,
                                                      decay=1e-6,
                                                      momentum=0.9,
                                                      nesterov=True),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train the Model like this
model_cifar.fit(CIFAR_X_train,
                CIFAR_y_train,
                epochs=15,
                batch_size=32,
                verbose=2,
                validation_split=0.2)

# Save the model
model_cifar.save('./model/mymodel.h5')

# Ramdom Predictions / Using Cifar
rand_25_idx = np.random.randint(0, 10000, size=25)
y_pred = model_cifar.predict(CIFAR_X_test[rand_25_idx])
y_pred = np.argmax(y_pred, axis=1).reshape(25, 1)
print("Test Accuracy: {}".format(np.sum(y_pred == CIFAR_y_test[rand_25_idx])/len(y_pred)))
