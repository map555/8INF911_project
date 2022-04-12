from DatasetBuilder import *
from Utils import *
from meme_generator import *
import tensorflow as tf

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1500)

if __name__ == '__main__':

    dict_test={"patate":69}
    print(dict_test["patate"])

    dict_test["clé"]="valeur"
    print(dict_test["clé"])


    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if isDatasetExist():
        df = importDataset()
    else:
        df = getDataset("ImgFlip575K_Dataset-master\\dataset\\memes")

    if IsMemesTextExist():
        memes_text = LoadMemesText()
    else:
        WriteMemesText(df)
        memes_text = LoadMemesText()

    # meme_generator = SimpleRNN(epoch_number=1, memes_text=memes_text)
    meme_generator2 = SimpleRNNWordByWord(epoch_number=1,memes_text=memes_text)

    for i in range(5):

        meme_generator2.generate_meme(50,100)

