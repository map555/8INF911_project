from DatasetBuilder import *
from Utils import *
from meme_generator import *
import tensorflow as tf

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1500)

if __name__ == '__main__':

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

    template_id_df = pd.read_excel("templates.xlsx",dtype={"template":str,"id":int})

    # meme_generator = SimpleRNN(epoch_number=1, memes_text=memes_text)
    #meme_generator2 = SimpleRNNWordByWord(epoch_number=30, memes_text=memes_text)
    # meme_generator = RNNWordByWordWithImage(epoch_number=1)
    meme_generator3 = RnnWordByWordWithImage2(epoch_number=2)

    for i in range(5):
        meme_generator3.GenerateMeme()

    #for i in range(5):
        #meme_generator2.generate_meme()
