from os.path import exists
import tensorflow as tf


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
