import collections
import json
import random
import time

import numpy

from DatasetBuilder import isDatasetExist, importDataset, getDataset
from ResNet.ResNetModel import ResNetModel
from sequence_generator import *
from RNN import *
from one_step import *
import os
import requests
import urllib
from datetime import datetime
from Utils import *
import pickle as pkl
from CNN_encoder import CNN_Encoder
from RNN_decoder import RNN_Decoder
from PIL import Image





# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load('image_text_example/memes/img/' + img_name.decode('utf-8') + '.jpg.npy')
    return img_tensor, cap


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs,
                                    r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")


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

        self.dataset = self.sequence_generator.generate_sequences(memes_text)
        self.model = RNNModel(vocab_size=len(self.sequence_generator.ids_from_chars.get_vocabulary()),
                              embedding_dim=self._embeddingDim, rnn_units=self._rnnUnits)

        if IsModelExist(model_name="model"):
            self.model.load_weights("model/model")
        else:
            self._train_model()

    def _train_model(self):
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)

        self.model.fit(self.dataset, epochs=self._epoch)

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
        self._embeddingDim = 1024
        self._rnnUnits = 512

        self.dataset = self.sequence_generator.generateSequences()
        self.model = RNNModel(vocab_size=len(self.sequence_generator.ids_from_words.get_vocabulary()),
                              embedding_dim=self._embeddingDim, rnn_units=self._rnnUnits)

        if IsModelExist(model_name="word_by_word_model"):
            self.model.load_weights("word_by_word_model/word_by_word_model")
        else:
            self._train_model()

    def _train_model(self):

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss, metrics=['acc'])

        self.model.fit(self.dataset, epochs=self._epoch)

        self.model.save_weights("word_by_word_model/word_by_word_model")

    def _generate_text(self, box_count):
        x = self.sequence_generator.ids_from_words
        one_step_model = OneStepWorldByWorld(self.model, self.sequence_generator.words_from_ids,
                                             self.sequence_generator.ids_from_words)

        states = None
        next_words = tf.constant([' '])
        initial_next_char = next_words
        meme_strings = []

        for i in range(box_count):
            next_words = initial_next_char
            string_result = ""
            while ('.' not in string_result and ',' not in string_result):
                next_words, states = one_step_model.generate_one_step(next_words, states=states)
                string_result += tf.strings.join(next_words).numpy().decode("utf-8") + " "
            string_result = string_result.replace('.', '')
            string_result = string_result.upper()
            meme_strings.append(string_result)

        return meme_strings

    def generate_meme(self):

        meme_id_df = self._getMemeIDList()

        random_meme_id = meme_id_df.iloc[random.randint(0, len(meme_id_df) - 1)]

        meme_text = self._generate_text(
            box_count=random_meme_id['box_count'])

        URL = 'https://api.imgflip.com/caption_image'

        params = {
            'username': self.__username,
            'password': self.__password,
            'template_id': random_meme_id['id']
        }

        for i in range(0, random_meme_id['box_count']):
            params['boxes[' + str(i) + '][text]'] = meme_text[i]

        response = requests.request('POST', URL, params=params).json()
        print(response)

        # Save the meme
        opener = urllib.request.URLopener()
        opener.addheader('User-Agent', self.__user_agent)
        filename, headers = opener.retrieve(response['data']['url'], 'ml_generated_meme_' + datetime.now().strftime(
            "%d-%b-%Y-%H-%M-%S-%f") + ".jpg")

    def _getMemeIDList(self):
        # Fetch 100 available memes
        data = requests.get('https://api.imgflip.com/get_memes').json()['data']['memes']
        images = [{'id': image['id'], 'box_count': image['box_count']} for image in data]

        df = pd.DataFrame(images)

        return df


class RNNWordByWordWithImage:
    sequence_generator = []
    model = []

    def __init__(self, epoch_number):
        self.resnet_model = ResNetModel()
        self.memes_df = self._load_memes_df()
        self.sequence_generator = WordImageSequenceGenerator(self.memes_df)
        self.__username = os.environ.get('meme_username')
        self.__password = os.environ.get('meme_password')
        self.__user_agent = os.environ.get("meme_user_agent")
        self._epoch = epoch_number
        self._embeddingDim = 8
        self._rnnUnits = 16

        self.sequence_generator.generate_sequences()  # TODO
        self.model = RNNModel(vocab_size=len(self.sequence_generator.ids_from_words.get_vocabulary()),
                              embedding_dim=self._embeddingDim, rnn_units=self._rnnUnits)

        if IsModelExist(model_name="word_by_word_with_image_model"):
            self.model.load_weights("word_by_word_with_image_model/word_by_word_with_image_model")
        else:
            self._train_model()

    def _load_memes_df(self):
        if not IsMemesTextByTemplateCSVExist():
            self._create_csv_file()
        memes_text_df = LoadMemesTextByTemplateFromCSV()
        memes_text_df['resnet_prediction'] = memes_text_df.apply(
            lambda x: self.resnet_model.predict(
                'ImgFlip575K_Dataset-master/dataset/img/' + x['template'] + '/' + x['template'] + '.jpg'), axis=1)
        return memes_text_df

    def _create_csv_file(self):
        if isDatasetExist():
            df = importDataset()
        else:
            df = getDataset("ImgFlip575K_Dataset-master\\dataset\\memes")
        WriteMemesTextByTemplateToCSV(df)

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


class RnnWordByWordWithImage2:
    def __init__(self, epoch_number):
        self._batchSize = 64
        self._buffSize = 1000
        self._embeddingDim = 256
        self._units = 512
        self._featureShape = 2048
        self._attentionFeatureShape = 64
        self._maxLenght = 50  # Max word count for a caption.
        self._vocabSize = 5000  # Use the top 5000 words for a vocabulary
        self._skip = True
        self._epochNumber =epoch_number
        self._startEpoch=0

        self._InitializeTemplateDF()
        self._InitializeTrainCaption()
        self._InitializeImageFeatureExtractModel()

        if self._skip is not True:
            self._ImageFeaturExtract()

        self._trainCaptions = [str(element) for element in self._trainCaptions]
        self._captionDS = tf.data.Dataset.from_tensor_slices(self._trainCaptions)

        self._InitializeTokenizer()
        self._InitializeTokenizerVectorAndIndex()
        self._InitializeImgCapVector()
        self._InitializeTrainingAndValidationDS()

        self._capTrain = [numpy.resize(element, 50) for element in self._capTrain]
        self._numSteps = len(self._imgNameTrain) // self._batchSize

        self._IntializePredictionDS()

        self._encoder = CNN_Encoder(self._embeddingDim)
        self._decoder = RNN_Decoder(self._embeddingDim, self._units, self._tokenizer.vocabulary_size())
        self._optimizer = tf.keras.optimizers.Adam()
        self._lossObject = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self._checkPointPath = "./checpoints/train"
        self._checkPoint = tf.train.Checkpoint(encoder=self._encoder,decoder=self._decoder,optimizer=self._optimizer)
        self._checkPointManager = tf.train.CheckpointManager(self._checkPoint, self._checkPointPath, max_to_keep=5)

        if self._checkPointManager.latest_checkpoint:
            self._startEpoch = int(self._checkPointManager.latest_checkpoint.split("-")[-1])
            # restoring the latest checkpoint in checkpoint_path
            self._checkPoint.restore(self._checkPointManager.latest_checkpoint)

        self._Train()

    def _InitializeTemplateDF(self):
        self._templatesDF = pd.DataFrame()

        if not exists('memes_df.csv'):
            gen = (file for file in os.listdir('memes') if os.path.isfile('memes/' + file))

            for file in gen:
                df_temp = pd.read_json('memes/' + file)
                df_temp['template_name'] = file.replace('.json', '')
                df_temp['text'] = df_temp.apply(lambda row: '. '.join(row['boxes']), axis=1)
                df_temp['text'] = df_temp.apply(lambda row: ' '.join(('<start>', row['text'], '<end>')), axis=1)
                df_temp = df_temp.drop(['metadata', 'post', 'url', 'boxes'], axis=1)

                if self._templatesDF.empty:
                    self._templatesDF = df_temp

                else:
                    self._templatesDF = pd.concat([self._templatesDF, df_temp])

            self._templatesDF.to_csv('memes_df.csv', index=False)

        else:
            self._templatesDF = pd.read_csv('memes_df.csv')

        self._templatesDF = self._templatesDF.dropna()

    def _GetShuffledKey(self, dict_to_shuffle):
        shuffled_key = list(dict_to_shuffle.keys())
        random.shuffle(shuffled_key)
        return shuffled_key

    def _InitializeTrainCaption(self):
        if not exists("image_path_to_caption.json"):
            image_path_to_caption = collections.defaultdict(list)

            for index, row in self._templatesDF.iterrows():
                image_path_to_caption[row['template_name']].append(row['text'])

            with open('image_path_to_caption.json', 'w') as fp:
                json.dump(image_path_to_caption, fp)

        else:
            with open('image_path_to_caption.json') as json_file:
                image_path_to_caption = json.load(json_file)

        self._trainCaptions = []
        self._imgNameVector = []

        for image_path in self._GetShuffledKey(image_path_to_caption):
            caption_list = image_path_to_caption[image_path]
            self._trainCaptions.extend(caption_list)
            self._imgNameVector.extend([image_path] * len(caption_list))

    def _InitializeImageFeatureExtractModel(self):
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

        # initialize model with input and hidden layer
        self._imageFeatureExtractor = tf.keras.Model(image_model.input, image_model.layers[-1].output)

    def _ImageFeaturExtract(self):
        # Get unique image
        encode = sorted(set(self._imgNameVector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode)
        image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

        for img, path in image_dataset:
            batch_features = self._imageFeatureExtractor(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy.decode("utf-8")
                np.save(path_of_feature, bf.numpy())

    def _InitializeTokenizer(self):
        if not exists("image_text_tokeniser.pkl"):
            self._tokenizer = tf.keras.layer.TextVectorization(max_tokens=self._vocabSize, standardize=standardize)

            # Learn the vocabulary from the caption data.
            self._tokenizer.adapt(self._captionDS)
            pkl.dump({"config": self._tokenizer.get_config(), "weights": self._tokenizer.get_weights()},
                     open("image_text_tokeniser.pkl", "wb"))

        else:
            tokenizer_info = pkl.load(open("image_text_tokeniser.pkl", "rb"))
            self._tokenizer = tf.keras.layers.TextVectorization.from_config(tokenizer_info["config"])
            self._tokenizer.set_weights(tokenizer_info["weights"])

    def _InitializeTokenizerVectorAndIndex(self):
        # Create the tokenized vectors
        self._tokenizerVector = self._captionDS.map(lambda x: self._tokenizer(x))

        # Create mappings for words to indices and indicies to words.
        self._wordToIndex = tf.keras.layers.StringLookup(mask_token="", vocabulary=self._tokenizer.get_vocabulary())
        self._indexToWord = tf.keras.layers.StringLookup(mask_token="", vocabulary=self._tokenizer.get_vocabulary(),
                                                         invert=True)

    def _InitializeImgCapVector(self):
        self._imgCapVector = collections.defaultdict(list)
        for img, cap in zip(self._imgNameVector, self._tokenizerVector):
            self._imgCapVector[img].append(cap)

    def _InitializeTrainingAndValidationDS(self):
        img_keys = self._GetShuffledKey(self._imgCapVector)
        slice_index = int(len(img_keys) * 0.8)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

        self._imgNameTrain = []
        self._capTrain = []
        self._imgNameVal = []
        self._capVal = []

        for imgt in img_name_train_keys:
            capt_len = len(self._imgCapVector[imgt])
            self._imgNameTrain.extend([imgt] * capt_len)
            self._capTrain.extend(self._imgCapVector[imgt])

        for imgv in img_name_val_keys:
            capv_len = len(self._imgCapVector[imgv])
            self._imgNameVal.extend([imgv] * capv_len)
            self._capVal.extend(self._imgCapVector[imgv])

    def _IntializePredictionDS(self):
        self._predictionDS = tf.data.Dataset.from_tensor_slices((self._imgNameTrain, self._capTrain))

        # Use map to load the numpy files in parallel
        self._predictionDS = self._predictionDS.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2],
                                                                                           [tf.float32, tf.int64]),
                                                    num_parallel_calls=tf.data.AUTOTUNE)

        # Shuffle and batch
        self._predictionDS = self._predictionDS.shuffle(self._buffSize).batch(self._batchSize)
        self._predictionDS = self._predictionDS.prefetch(buffer_size=tf.data.AUTOTUNE)

    def _Train(self):
        # adding this in a separate cell because if you run the training cell
        # many times, the loss_plot array will be reset
        loss_plot = []

        for ep in range(self._startEpoch,self._epochNumber):
            start = time.time()
            total_loss =0

            for (batch, (img_tensor,target)) in enumerate(self._predictionDS):
                batch_loss, t_loss = self._trainStep(img_tensor,target)
                total_loss+=t_loss

                if batch %100 == 0:
                    average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                    print(f'Epoch {ep + 1} Batch {batch} Loss {average_batch_loss:.4f}')
            # storing the epoch end loss value to plot later
            loss_plot.append(total_loss / self._numSteps)
            self._checkPointManager.save()
            print(f'Epoch {ep + 1} Loss {total_loss / self._numSteps:.6f}')
            print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')

    def GenerateMeme(self):
        template_id_df = pd.read_excel("templates.xlsx", dtype={"template": str, "id": int})

        template_row_id = self._RandomTemplateID(template_id_df.shape[0])
        template_data = template_id_df.iloc[[template_row_id]]
        template_id = template_data["id"].values[0]
        template_name = template_data["template"].values[0]
        templates_img_path = "image_text_example/memes_original/img/{img_name}.jpg".format(img_name=template_name)

        result, attention_plot = self._Evaluate(templates_img_path)

        self._SendMemeToImgFlip(template_id,result)


    @tf.function
    def _trainStep(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self._decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self._wordToIndex('<start>')] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self._encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self._decoder(dec_input, features, hidden)

                loss += self._LossFunction(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self._encoder.trainable_variables + self._decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self._optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss

    def _LossFunction(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self._lossObject(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def _RandomTemplateID(self,size):
        return random.randint(0,size-1)

    def _SendMemeToImgFlip(self, template_id, meme_texts):

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

    def _Evaluate(self, image):
        attention_plot = np.zeros((self._maxLenght, self._attentionFeatureShape))

        hidden = self._decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = self._imageFeatureExtractor(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                     -1,
                                                     img_tensor_val.shape[3]))

        features = self._encoder(img_tensor_val)

        dec_input = tf.expand_dims([self._wordToIndex('<start>')], 0)
        result = []
        result_temp = []
        meme_text_boxes = []

        for i in range(self._maxLenght):
            predictions, hidden, attention_weights = self._decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            predicted_word = tf.compat.as_text(self._indexToWord(predicted_id).numpy())
            if predicted_word == '[UNK]':
                continue

            if predicted_word == '<end>':
                meme_text_boxes.append(" ".join(result_temp))
                return meme_text_boxes, attention_plot
            elif predicted_word == "." or predicted_word.endswith("."):
                meme_text_boxes.append(" ".join(result_temp))
                result_temp = []  # empty result_temp
            else:
                result.append(predicted_word)
                result_temp.append(predicted_word)

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        meme_text_boxes.append(" ".join(result_temp))
        return meme_text_boxes, attention_plot