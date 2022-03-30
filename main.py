# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from DatasetBuilder import *

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1500)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if isDatasetExist():
        df = importDataset()
    else:
        df = getDataset("ImgFlip575K_Dataset-master\\dataset\\memes")
    print("\"Hello\"")
    print(df)

    meme_text = df['content'].str.cat()

    with open("meme_text.txt", "w", encoding='utf-8') as text_file:
        text_file.write(meme_text)

