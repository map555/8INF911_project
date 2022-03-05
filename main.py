# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from DatasetBuilder import *




# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    if isDatasetExist():
        df = importDataset()
    else:
        df = getDataset("ImgFlip575K_Dataset-master\\dataset\\memes")
    print()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
