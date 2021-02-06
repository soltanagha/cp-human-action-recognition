import os, sys
from tqdm import tqdm
from shutil import copyfile


def dataList(fileName=None):
    if fileName is None:
        sys.exit("File name is absent: [dataList(fileName=None)]")

    if validPath(fileName) is False:
        sys.exit("Provided file does not exist: [dataList(fileName=None)]")

    txtFile = open(fileName, "r")

    dataList = [dataList.strip("\n").split(" ")[0] for dataList in txtFile]

    print("File name:{}   Lines of data:{:,} ".format(fileName, len(dataList)))
    return dataList


def validPath(dirName=None):
    if dirName is None:
        sys.exit("Path is not availabe: [dirAvailabe(dirName=None)]".format())
    else:
        if not os.path.exists(dirName):
            return False
        else:
            return True


def duplicater(folder_vide_name, save_dir):
    base_path = "/Users/soltanagha/Documents/UCF101/"

    dataPath = str(base_path) + str(folder_vide_name)

    copyfile(dataPath, save_dir)
    print("Duplicated {} successfully".format(folder_vide_name))

type = int(input("'0' for Train Data and '1' for Test Data? "))
if type:
    textFileList = "testlist01.txt"
    saveDestination = "/ds/Test_Dataset"
else:
    textFileList = "trainlist01.txt"
    saveDestination = "/ds/Train_Dataset"

ls = dataList(textFileList)

skips = []

for _ls in tqdm(ls):

    nFolder_, nFile = os.path.split(_ls)
    nFolder = os.path.join(saveDestination, nFolder_)
    nFile_ = os.path.join(nFolder, nFile)
    nFolderFile = os.path.join(nFolder_, nFile)

    if not os.path.exists(nFolder):
        os.makedirs(nFolder)

    if not os.path.exists(nFile_):
        print("Duplicating file to defined directory: ", nFile)
        if nFile not in skips:
            duplicater(nFolderFile, nFile_)
    else:
        print("Skipped: ", nFile_)
