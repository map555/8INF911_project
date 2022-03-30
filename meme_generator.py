from sequence_generator import *
from RNN import *
from one_step import OneStep


class MemeGenerator:
    sequence_generator = []
    model = []

    def __init__(self):
        self.sequence_generator = SequenceGenerator()

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

    def generate_meme(self):
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
