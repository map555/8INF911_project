import tensorflow as tf


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
    def __init__(self,text):
        self._text = text

    def generateSequences(self):
        # generate sequences
        # length of text is the number of characters in it

        # Take a look at the first 250 characters in text

        # The unique characters in the file
        t = self._text.lower()

        t = t.replace("!"," ")
        t = t.replace("\""," ")
        t = t.replace("\'"," ")
        t = t.replace("#"," ")
        t = t.replace("$"," ")
        t = t.replace("%"," ")
        t = t.replace("&"," ")
        t = t.replace("@"," ")
        t = t.replace("\\"," ")
        t = t.replace("\n", " ")

        self._text=t



        vocab = sorted(set(t.split(" ")))

        self.ids_from_words = tf.keras.layers.StringLookup(
            vocabulary=list(vocab), mask_token=None)

        self.words_from_ids = tf.keras.layers.StringLookup(
            vocabulary=self.ids_from_words.get_vocabulary(), invert=True, mask_token=None)

        all_ids = self.ids_from_words(tf.strings.unicode_split(self._text, 'UTF-8'))

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

