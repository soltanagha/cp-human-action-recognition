from preprocessor import *
import pickle

def save_structure(structure, name):
    with open(base_path + 'Structures/' + name + '.pickle', 'wb') as handle:
        pickle.dump(structure, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_structure(name):
    if os.path.isfile(base_path + 'Structures/' + name + '.pickle'):
        with open(base_path + 'Structures/' + name + '.pickle', 'rb') as handle:
            structure = pickle.load(handle)
        return structure
    else:
        return []

def get_class(value):
    value = get_value(value)
    for class_name in classes:
        if classes[class_name] == value:
            return class_name


def shuffle(X_data, y_data):
    X_data_series = pd.Series(X_data)
    y_data_series = pd.Series(y_data)

    dataFrame = pd.DataFrame()
    dataFrame = pd.concat([X_data_series, y_data_series], axis=1)

    dataArray = np.array(dataFrame)
    np.random.shuffle(dataArray)

    return dataArray[:, 0], dataArray[:, 1]


def shuffleTestData(testingData_paths):
    halfOfTestData = int(len(testingData_paths) / 2)
    a, b = shuffle(testingData_paths[:halfOfTestData], testingData_paths[halfOfTestData:])
    return np.concatenate((a, b), axis=0)

def get_value(value):
    if type(value) == type([]) or type(value) == type(np.array([])):
        return np.argmax(value)
    return value

def get_test_data_names(test_path):
    data_names = []
    for file in tqdm(os.listdir(test_path)):
        data_names.append(file)

    return data_names