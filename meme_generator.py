from sequence_generator import *
from RNN import *
from one_step import OneStep
import os
import requests
import urllib


class MemeGenerator:
    sequence_generator = []
    model = []

    def __init__(self):
        self.sequence_generator = SequenceGenerator()
        self.__username = os.environ.get('meme_username')
        self.__password = os.environ.get('meme_password')
        self.__user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36'

    def train_model(self, memes_text):
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

        EPOCHS = 1

        self.model.fit(dataset, epochs=EPOCHS)

        self.model.save('model/')

    def generate_text(self):
        one_step_model = OneStep(self.model, self.sequence_generator.chars_from_ids,
                                 self.sequence_generator.ids_from_chars)

        states = None
        next_char = tf.constant([' '])
        result = [next_char]

        for n in range(100):
            next_char, states = one_step_model.generate_one_step(next_char, states=states)
            result.append(next_char)

        result = tf.strings.join(result)

        print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)

    def generate_meme(self):

        # Fetch the available memes
        data = requests.get('https://api.imgflip.com/get_memes').json()['data']['memes']
        images = [{'name': image['name'], 'url': image['url'], 'id': image['id']} for image in data]

        # List all the memes
        print('Here is the list of available memes : \n')
        ctr = 1
        for img in images:
            print(ctr, img['id'], img['name'])
            ctr = ctr + 1

        URL = 'https://api.imgflip.com/caption_image'
        params = {
            'username': self.__username,
            'password': self.__password,
            'template_id': 181913649,
            'text0': 'text0',
            'text1': 'text1'
        }
        response = requests.request('POST', URL, params=params).json()
        print(response)

        # Save the meme
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', self.__user_agent)
        filename, headers = opener.retrieve(response['data']['url'], '2.jpg')

