from DatasetBuilder import *
from Utils import *
from meme_generator import *

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1500)

if __name__ == '__main__':

    if isDatasetExist():
        df = importDataset()
    else:
        df = getDataset("ImgFlip575K_Dataset-master\\dataset\\memes")

    if IsMemesTextExist():
        memes_text = LoadMemesText()
    else:
        WriteMemesText(df)
        memes_text = LoadMemesText()

    meme_generator = MemeGenerator(epoch_number=1,memes_text=memes_text)

    for i in range(5):

        meme_generator.generate_meme(50,100)
