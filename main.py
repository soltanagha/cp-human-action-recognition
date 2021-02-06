from model_implementation import *

load_tracker = {"CNN_3D_Model":"n",
                "train_data":"n",
                "test_data":"n",
                }

for file_name in load_tracker:
    user_input = input("Load " + file_name + " y/n? : ")
    load_tracker[file_name] = user_input

print('\n', load_tracker)

if load_tracker["train_data"] == "n":
    print("Reading Training Data..")
    trainingData_paths, y_train = read_data_paths(train_path, True)
    trainingData_paths, y_train = shuffle(trainingData_paths, y_train)

if load_tracker["train_data"] == "n":
    display_data(trainingData_paths,"train_data")

if load_tracker["test_data"] == "n":
    print("Reading Testing Data..")
    testingData_paths = read_data_paths(test_path, False)
    testingData_paths = shuffleTestData(testingData_paths)

no_test_data = int(input("Currently uploaded " +str(len(testingData_paths))+" test video.Enter test data size : "))

if load_tracker["test_data"] == "n":
    display_data(testingData_paths[:no_test_data],"test_data")

if load_tracker["train_data"] == "n":
    X_train, y_train = preprocess(trainingData_paths, y_train)
    save_structure(X_train, "X_train")
    save_structure(y_train, "y_train")
else:
    X_train = load_structure("X_train")
    y_train = load_structure("y_train")
    print("Training data is Loaded!")

#Test Data Preprocessing
if load_tracker["test_data"] == "n":
    X_test, _ = preprocess(testingData_paths[:no_test_data], [])
    #save_structure(X_test, "X_test")
else:
    X_test = load_structure("X_test")
    print("Testing data is Loaded")

#Create/Load Model and Show it's Structure
if load_tracker["CNN_3D_Model"] == "n":
   create_model()
   save_model_plot(model)
   print("Model is created!")
else:
    load_model(0)
    print("Model is loaded!")
"""
train_model = input("Train model with validation 0.2 y/n? : ")
if train_model == 'y':
    train(X_train, y_train, val_split=0.2)
    show_model_performance()

train_model = input("Train model on all data y/n? : ")
if train_model == 'y':
    train(X_train, y_train, val_split=0)
    show_model_performance()"""

test_model = input("Test model y/n? : ")
if test_model == 'y':
    y_predict = predict(X_test)
    have_a_look(X_test, y_predict)
