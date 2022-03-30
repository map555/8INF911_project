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

    meme_generator = MemeGenerator()

    meme_generator.train_model(memes_text)

    meme_generator.generate_meme()
