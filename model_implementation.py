from preprocessor import *
from visualization import *
import keras
from keras import layers
from keras import Model
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, ZeroPadding3D


# Define model
model = None

CNN_type  = 0

weights_path = base_path + "Weights/sports1M_weights_tf.h5"

def create_model():
    global model
    create_3D_CNN_model()
    compile_model()
    model.save(base_path + "Models/CNN_3D_Model")

    print(model.summary())

def compile_model():
    global model
    # Compile the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

def load_model(model_type):
    global model
    model = keras.models.load_model(base_path + "Models/CNN_3D_Model")
    print(model.summary())


# Create 3D CNN model
def create_3D_CNN_model():
    global model
    model = Sequential(name="3D-CNN Model")

    # 1st layer
    model.add(Conv3D(64, (3, 3, 3), activation="relu", name="conv1",
                     input_shape=(D, W, H, C),
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

    # 2nd layer
    model.add(Conv3D(128, (3, 3, 3), activation="relu", name="conv2",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

    # 3rd layer
    model.add(Conv3D(256, (3, 3, 3), activation="relu", name="conv3a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(256, (3, 3, 3), activation="relu", name="conv3b",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

    # 4th layer
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv4a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv4b",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))

    # 5th layer
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv5a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv5b",
                     strides=(1, 1, 1), padding="same"))

    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(101, activation='softmax', name='fc8'))

    if weights_path:
        model.load_weights(weights_path)

    model.layers.pop()
    pre_last_layer_output = model.layers[-1].output
    last_layer_output = Dense(5, activation='softmax', name='fc9')(pre_last_layer_output)

    model = Model(model.input, last_layer_output)

    for layer in model.layers[:-5]:
        layer.trainable = False

batch_size = 50
no_epochs = 50
learning_rate = 0.0001
validation_split = 0.2
verbosity = 1

def train(X_train, y_train, val_split):
    global model
    global validation_split
    validation_split = val_split

    # Convert target vectors to categorical targets
    y_train = to_categorical(y_train).astype(np.integer)

    # Fit data to model
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=no_epochs,
                        verbose=verbosity,
                        validation_split=val_split)


def test(X_test, y_test):
    # Convert target vectors to categorical targets
    y_test = to_categorical(y_test).astype(np.integer)

    # Generate generalization metrics
    model_loss, model_accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)

    return model_loss, model_accuracy, y_pred

def predict(X_test):
    y_pred = model.predict(X_test)
    return y_pred
