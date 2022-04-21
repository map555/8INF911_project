import tensorflow as tf
import re


class SequenceGenerator:
    ids_from_chars = []
    chars_from_ids = []

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text

    def generate_sequences(self, memes_text):
        # generate sequences
        # length of text is the number of characters in it

        # Take a look at the first 250 characters in text

        # The unique characters in the file
        vocab = sorted(set(memes_text))

        self.ids_from_chars = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)

        self.chars_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

        all_ids = self.ids_from_chars(tf.strings.unicode_split(memes_text, 'UTF-8'))

        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        seq_length = 100
        examples_per_epoch = len(memes_text) // (seq_length + 1)

        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        dataset = (
            dataset
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

        return dataset


class WordSequenceGenerator:
    def __init__(self, text):
        self._text = text

    def generateSequences(self):
        # generate sequences
        # length of text is the number of characters in it

        # Take a look at the first 250 characters in text

        # The unique characters in the file
        t = self._text.upper()

        t = t.replace("!", "")
        t = t.replace("\"", "")
        t = t.replace("\'", "")
        t = t.replace("#", "")
        t = t.replace("$", "")
        t = t.replace("%", "")
        t = t.replace("&", "")
        t = t.replace("@", "")
        t = t.replace("\\", "")
        t = re.sub(r'[^a-zA-Z\n]', ' ', t)
        t = re.sub(r' +', ' ', t)
        t = t.replace("\n\n", ". ") # maybe remove all dots?
        t = t.replace("\n", ". ")
        t = t.replace(" .", ".")

        self._text = t

        vocab = sorted(set(self._text.split(" ")))

        self.ids_from_words = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)

        self.words_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_words.get_vocabulary(), invert=True, mask_token=None)

        all_ids = self.ids_from_words(tf.strings.split(self._text, sep=" "))

        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        seq_length = 35

        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        print(dataset)

        dataset = (
            dataset
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text


class WordImageSequenceGenerator:
    def __init__(self, memes_text_df):
        self._memes_text_df = memes_text_df

    def generate_sequences(self):
        # generate sequences
        # length of text is the number of characters in it

        # Take a look at the first 250 characters in text

        # The unique characters in the file

        print(self._memes_text_df.columns.tolist())

        print(self._memes_text_df['text'])

        self._memes_text_df['text'] = self._memes_text_df.apply(lambda row: row['text'].lower(), axis=1)
        self._memes_text_df['text'] = self._memes_text_df.apply(lambda row: row['text'].replace('\\', ''), axis=1)
        self._memes_text_df['text'] = self._memes_text_df.apply(lambda row: re.sub(r'[^a-zA-Z]', ' ', row['text']),
                                                                axis=1)
        self._memes_text_df['text'] = self._memes_text_df.apply(lambda row: re.sub(r' +', ' ', row['text']),
                                                                axis=1)
        self._memes_text_df['text'] = self._memes_text_df.apply(lambda row: row['text'].replace("\n", " "), axis=1)

        all_text = self._memes_text_df['text'].apply(lambda row: ''.join(row))

        vocab = sorted(set(all_text.split(" ")))

        self.ids_from_words = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)

        self.words_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_words.get_vocabulary(), invert=True, mask_token=None)

        print(tf.strings.split(self._text, sep=" "))

        all_ids = self.ids_from_words(tf.strings.split(self._text, sep=" "))

        print(all_ids)

        ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

        seq_length = 100

        sequences = ids_dataset.batch(seq_length + 1, drop_remainder=True)

        dataset = sequences.map(self.split_input_target)

        # Batch size
        BATCH_SIZE = 64

        # Buffer size to shuffle the dataset
        # (TF data is designed to work with possibly infinite sequences,
        # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
        # it maintains a buffer in which it shuffles elements).
        BUFFER_SIZE = 10000

        print(dataset)

        dataset = (
            dataset
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(tf.data.experimental.AUTOTUNE))

        return dataset

    def split_input_target(self, sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
