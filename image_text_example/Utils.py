import os
import urllib
from os.path import exists, isdir, join
from os import getcwd

import requests
import tensorflow as tf
from tensorflow import keras as kr
import pandas as pd
from datetime import datetime

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


def IsMemesTextByTemplateCSVExist():
    return exists('memes_text_by_template.csv')


def WriteMemesTextByTemplateToCSV(df):
    group = df.groupby('template')

    df = group.apply(lambda x: x['content'].str.cat(sep='\n\n'))

    df.to_csv('memes_text_by_template.csv')


def LoadMemesTextByTemplateFromCSV():
    df = pd.read_csv('memes_text_by_template.csv')

    df = df.rename(columns={"0": "text"})

    return df


def IsModelExist(model_name):
    return isdir(join(getcwd(), model_name))


def LoadModel():
    return kr.models.load_model("model", custom_objects={"RNNModel": RNN.RNNModel})

def SendMemeToImgFlip(template_id,meme_texts):

    URL = 'https://api.imgflip.com/caption_image'

    params = {
        'username': os.environ.get('meme_username'),
        'password': os.environ.get('meme_password'),
        'template_id': template_id
    }

    for i in range(len(meme_texts)):
        params['boxes[' + str(i) + '][text]'] = meme_texts[i]

    response = requests.request('POST', URL, params=params).json()
    print(response)

    # Save the meme
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', os.environ.get("meme_user_agent"))
    filename, headers = opener.retrieve(response['data']['url'], 'ml_generated_meme_' + datetime.now().strftime(
        "%d-%b-%Y-%H-%M-%S-%f") + ".jpg")
