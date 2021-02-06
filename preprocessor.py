import os
from tqdm import tqdm
import cv2 as cv
import numpy as np
import pandas as pd


base_path = "ds/"
train_path = base_path + "Train_Dataset"
test_path = base_path + "Test_Dataset"
classes = {'Diving':0, 'Bowling':1, 'Basketball':2, 'TennisSwing':3,  'PoleVault':4}
no_classes = len(classes)

def read_data_paths(path,istruth):
    data_paths = []
    truth = []

    for class_name in classes:
        for file in tqdm(os.listdir(path + "/" + class_name)):
            if not file.startswith('.'):
                vid_path = os.path.join(path + "/" + class_name, file)
                data_paths.append(vid_path)
                truth.append(classes[class_name])

    if istruth:
        return data_paths, truth
    else:
        return data_paths


def get_test_data_names():
    data_names = []
    for file in tqdm(os.listdir(test_path)):
        data_names.append(file)

    return data_names

def shuffle(X_data, y_data):
    X_data_series = pd.Series(X_data)
    y_data_series = pd.Series(y_data)

    dataFrame = pd.DataFrame()
    dataFrame = pd.concat([X_data_series, y_data_series], axis=1)

    dataArray = np.array(dataFrame)
    np.random.shuffle(dataArray)

    return dataArray[:, 0], dataArray[:, 1]

D = 16   #New Depth size => Number of frames.
W = 112  #New Frame Width.
H = 112  #New Frame Height.
C = 3    #Number of channels.
sample_shape = (D, W, H, C) #Single Video shape.


def preprocess(data_paths, data_truth):
    all_videos = []

    for i in tqdm(range(len(data_paths))):
        cap = cv.VideoCapture(data_paths[i])

        single_video_frames = []
        while (True):
            read_success, current_frame = cap.read()

            if not read_success:
                break

            current_frame = cv.resize(current_frame, (W, H))
            single_video_frames.append(current_frame)

        cap.release()

        single_video_frames = np.array(single_video_frames)
        single_video_frames.resize((D, W, H, C))

        all_videos.append(single_video_frames)

    all_videos = np.array(all_videos)
    data_truth = np.array(data_truth)

    return all_videos, data_truth