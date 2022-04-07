from os.path import exists, isdir, join
from os import getcwd
import tensorflow as tf
from tensorflow import keras as kr

import RNN


def IsMemesTextExist():
    dataset_file_existance = exists("meme_text.txt")

    return dataset_file_existance


def WriteMemesText(df):
    meme_text = df['content'].str.cat(sep="\n\n")
    with open("meme_text.txt", "w", encoding='utf-8') as text_file:
        text_file.write(meme_text)
        text_file.close()


def LoadMemesText():
    memes_text = ""
    with open("meme_text.txt", "r", encoding="utf-8") as text_file:
        memes_text = text_file.read()
        text_file.close()

    return memes_text

def IsModelExist(model_name):
    return isdir(join(getcwd(),model_name))

def LoadModel():
    return kr.models.load_model("model", custom_objects={"RNNModel":RNN.RNNModel})


