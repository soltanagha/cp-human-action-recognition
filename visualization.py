from helper import *
from matplotlib import pyplot as plt
from keras.utils import plot_model
import numpy as np

def display_data(data_paths,type):
    vids_shape = []
    for vid_path in tqdm(data_paths):
        cap = cv.VideoCapture(vid_path)

        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        depth = cap.get(cv.CAP_PROP_FRAME_COUNT)

        vids_shape.append((width, height, depth))

    print("\nWe have", len(vids_shape), " videos.")

    if type == "train_data":
        data_counter = {'Diving': 0, 'Bowling': 0, 'Basketball': 0, 'TennisSwing': 0, 'TaiChi': 0}
        for class_name in classes:
            data_counter[class_name] = len(os.listdir(train_path + "/" + class_name))

        print("Videos are divided into ", data_counter)

    print("Videos Shapes are:")
    data_frame = pd.Series(vids_shape).value_counts()
    print(data_frame.head(len(data_frame)))

def have_a_look(X_data, y_data):
    plt.figure(figsize=(20, 20))
    print(len(X_data))
    for n, i in enumerate(list(np.random.randint(0, len(X_data), 9))):  # Pick random 36 videos
        plt.subplot(3, 3, n+1)
        plt.axis('off')

        label = y_data[i]  # ex-> label = [0.2, 0.3, 0.2, 0.8, 0.6]
        plt.title(get_class(get_value(label)))  # The highest value is 0.8 which is at class no. 4

        first_frame = X_data[i][0]  # Pick first frame of this video.
        if C == 1:
            first_frame = first_frame.reshape((W, H))
        plt.imshow(first_frame)
    plt.show()


def plot_model_metrics(history):
    plt.figure(figsize=(8, 4))
    plt.title('Model Performance for Video Action Detection', size=18, c="C7")
    plt.ylabel('Accuracy value', size=15, color='C7')
    plt.xlabel('Epoch No.', size=15, color='C7')
    # plt.plot(history.history['loss'],  'o-', label='Training Data Loss', linewidth=2, c='C3') #C3 = red color.
    plt.plot(history.history['accuracy'],  'o-', label='Training Data Accuracy', linewidth=2, c='C2')  # C2 = green color.

    if len(history.history) > 2:
        plt.plot(history.history['val_accuracy'],  'o-', label='Validation Data Accuracy', linewidth=2, c='b')  # b = blue color.

    plt.legend()
    plt.savefig('model_performance.png')
    plt.show()

def show_model_performance(model):
    plot_model_metrics(model.history)

def save_model_plot(name,model):
    plot_model(model, to_file=name+'.png', show_shapes=True)
