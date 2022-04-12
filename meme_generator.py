import random

from sequence_generator import *
from RNN import *
from one_step import *
import os
import requests
import urllib
from datetime import datetime
from Utils import IsModelExist, LoadModel


class SimpleRNN:
    sequence_generator = []
    model = []

    def __init__(self, epoch_number, memes_text):
        self.sequence_generator = SequenceGenerator()
        self.__username = os.environ.get('meme_username')
        self.__password = os.environ.get('meme_password')
        self.__user_agent = os.environ.get("meme_user_agent")
        self._epoch = epoch_number
        self._embeddingDim = 512
        self._rnnUnits = 1024

        self.sequence_generator.generate_sequences(memes_text)
        self.model = RNNModel(vocab_size=len(self.sequence_generator.ids_from_chars.get_vocabulary()),
                              embedding_dim=self._embeddingDim, rnn_units=self._rnnUnits)

        if IsModelExist(model_name="model"):
            self.model.load_weights("model/model")
        else:
            self._train_model(memes_text)

    def _train_model(self, memes_text):
        dataset = self.sequence_generator.generate_sequences(memes_text)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)

        self.model.fit(dataset, epochs=self._epoch)

        self.model.save_weights("model/model")

    def _generate_text(self, min_text_lenght, max_text_lenght):
        x = self.sequence_generator.ids_from_chars
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


class SimpleRNNWordByWord:
    sequence_generator = []
    model = []

    def __init__(self, epoch_number, memes_text):
        self.sequence_generator = WordSequenceGenerator(memes_text)
        self.__username = os.environ.get('meme_username')
        self.__password = os.environ.get('meme_password')
        self.__user_agent = os.environ.get("meme_user_agent")
        self._epoch = epoch_number
        self._embeddingDim = 8
        self._rnnUnits = 16

        self.sequence_generator.generateSequences()
        self.model = RNNModel(vocab_size=len(self.sequence_generator.ids_from_words.get_vocabulary()),
                              embedding_dim=self._embeddingDim, rnn_units=self._rnnUnits)

        # print(self.model.summary())

        if IsModelExist(model_name="word_by_word_model"):
            self.model.load_weights("word_by_word_model/word_by_word_model")
        else:
            self._train_model()

    def _train_model(self):
        dataset = self.sequence_generator.generateSequences()

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)

        self.model.fit(dataset, epochs=self._epoch)

        self.model.save_weights("word_by_word_model/word_by_word_model")

    def _generate_text(self, min_text_lenght, max_text_lenght):
        x = self.sequence_generator.ids_from_words
        one_step_model = OneStepWorldByWorld(self.model, self.sequence_generator.words_from_ids,
                                             self.sequence_generator.ids_from_words)

        states = None
        next_words = tf.constant([' '])
        initial_next_char = next_words
        meme_strings = []

        for i in range(2):
            next_words = initial_next_char
            string_result = ""
            while (len(string_result) < max_text_lenght):
                next_words, states = one_step_model.generate_one_step(next_words, states=states)
                string_result += tf.strings.join(next_words).numpy().decode("utf-8") + " "
            meme_strings.append(string_result)

        return meme_strings

    def generate_meme(self, min_text_lenght, max_text_lenght):
        meme_text = self._generate_text(min_text_lenght=min_text_lenght, max_text_lenght=max_text_lenght)

        meme_id_list = self._getMemeIDList()

        random_meme_id = meme_id_list[random.randint(0, len(meme_id_list) - 1)]

        URL = 'https://api.imgflip.com/caption_image'

        params = {
            'username': self.__username,
            'password': self.__password,
            'template_id': random_meme_id,
            'boxes[0][text]': meme_text[0],
            'boxes[1][text]': meme_text[1]
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
