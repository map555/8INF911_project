import random

from sequence_generator import *
from RNN import *
from one_step import OneStep
import os
import requests
import urllib
from datetime import datetime
from Utils import IsModelExist,LoadModel


class MemeGenerator:
    sequence_generator = []
    model = []

    def __init__(self, epoch_number, memes_text):
        self.sequence_generator = SequenceGenerator()
        self.__username = os.environ.get('meme_username')
        self.__password = os.environ.get('meme_password')
        self.__user_agent = os.environ.get("meme_user_agent")
        self._epoch = epoch_number
        if IsModelExist():
            self.model=LoadModel()
            self.sequence_generator.generate_sequences(memes_text)
        else:
            self._train_model(memes_text)

    def _train_model(self, memes_text):
        dataset = self.sequence_generator.generate_sequences(memes_text)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        self.model = RNNModel(
            # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=len(self.sequence_generator.ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)

        self.model.fit(dataset, epochs=self._epoch)

        self.model.save('model/')

    def _generate_text(self, min_text_lenght, max_text_lenght):
        x=self.sequence_generator.ids_from_chars
        one_step_model = OneStep(self.model, self.sequence_generator.chars_from_ids,
                                 self.sequence_generator.ids_from_chars)

        states = None
        next_char = tf.constant([' '])
        initial_next_char = next_char
        result = [next_char]

        next_line_counter = 0
        while (len(result) < min_text_lenght) and (next_line_counter != 2):
            result = [initial_next_char]
            next_line_counter = 0
            for n in range(max_text_lenght):
                next_char, states = one_step_model.generate_one_step(next_char, states=states)
                if tf.strings.join(next_char) == "\n":
                    next_line_counter += 1

                if next_line_counter == 2:
                    continue
                else:
                    result.append(next_char)

        result = tf.strings.join(result)

        meme_text = result[0].numpy().decode('utf-8')
        return meme_text

    def generate_meme(self, min_text_lenght, max_text_lenght):
        meme_text = self._generate_text(min_text_lenght=min_text_lenght, max_text_lenght=max_text_lenght)
        meme_text_separator_index = meme_text.find("\n")
        meme_text_line1 = meme_text[0:meme_text_separator_index]
        meme_text_line2 = meme_text[meme_text_separator_index + 1:len(meme_text)]

        meme_id_list = self._getMemeIDList()

        random_meme_id = meme_id_list[random.randint(0, len(meme_id_list) - 1)]

        URL = 'https://api.imgflip.com/caption_image'
        params = {
            'username': self.__username,
            'password': self.__password,
            'template_id': random_meme_id,
            'text0': meme_text_line1,
            'text1': meme_text_line2
        }
        response = requests.request('POST', URL, params=params).json()
        print(response)

        # Save the meme
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', self.__user_agent)
        filename, headers = opener.retrieve(response['data']['url'], 'ml_generated_meme_' + datetime.now().strftime(
            "%d-%b-%Y-%H-%M-%S-%f") + ".jpg")

    def _getMemeIDList(self):
        # Fetch the available memes
        data = requests.get('https://api.imgflip.com/get_memes').json()['data']['memes']
        images = [{'name': image['name'], 'url': image['url'], 'id': image['id']} for image in data]

        ids = []
        for img in images:
            ids.append(img["id"])

        return ids
