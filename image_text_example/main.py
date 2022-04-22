import collections
import json
import random
import time

import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle as pkl

from PIL import Image

from CNN_encoder import CNN_Encoder
from RNN_decoder import RNN_Decoder
from Utils import *


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


# Load the numpy files
def map_func(img_name, cap):
    img_tensor = np.load('memes/img/' + img_name.decode('utf-8') + '.jpg.npy')
    return img_tensor, cap


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []
    result_temp = []
    meme_text_boxes = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
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


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result / 2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def RandomTemplateID(size):
    return random.randint(0, size - 1)


def GetImageTextDF():
    df = pd.DataFrame()

    gen = (file for file in os.listdir('memes') if os.path.isfile('memes/' + file))
    for file in gen:

        df_temp = pd.read_json('memes/' + file)

        df_temp['template_name'] = file.replace('.json', '')

        df_temp['text'] = df_temp.apply(lambda row: '. '.join(row['boxes']), axis=1)

        df_temp['text'] = df_temp.apply(lambda row: ' '.join(('<start>', row['text'], '<end>')), axis=1)

        df_temp = df_temp.drop(['metadata', 'post', 'url', 'boxes'], axis=1)

        if df.empty:
            df = df_temp
        else:
            df = pd.concat([df, df_temp])

    df.to_csv('memes_df.csv', index=False)

    df = pd.read_csv('memes_df.csv')

    df = df.dropna()

    return df


def GetTemplatesToCaption(df=None):
    if df is None:
        with open('image_path_to_caption.json') as json_file:
            image_path_to_caption = json.load(json_file)


    else:
        image_path_to_caption = collections.defaultdict(list)

        for index, row in df.iterrows():
            image_path_to_caption[row['template_name']].append(row['text'])

        with open('image_path_to_caption.json', 'w') as fp:
            json.dump(image_path_to_caption, fp)

    return image_path_to_caption


def GetImagePath(df):
    image_path_to_caption = collections.defaultdict(list)


def GetShuffledKeys(dict_to_shuffle):
    shuffled_keys = list(dict_to_shuffle.keys())
    random.shuffle(shuffled_keys)
    return shuffled_keys


def GetTrainCaptions(df=None):
    if df is None:
        image_path_to_caption = GetTemplatesToCaption()
    else:
        image_path_to_caption = GetTemplatesToCaption(df)

    train_captions = []
    img_name_vector = []

    for image_path in GetShuffledKeys(image_path_to_caption):
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    return train_captions, img_name_vector


def GetInitializedImageModel():
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

    # return model with input and hidden layer setted
    return tf.keras.Model(image_model.input, image_model.layers[-1].output)


def GetImageTextTokenizer(caption_dataset=None, max_caption_length=None, vocab_size=None):
    if caption_dataset is not None:
        tokenizer = tf.keras.layers.TextVectorization(max_tokens=vocab_size, standardize=standardize,
                                                      output_sequence_length=max_caption_length)
        # Learn the vocabulary from the caption data.
        tokenizer.adapt(caption_dataset)
        pkl.dump({"config": tokenizer.get_config(), "weights": tokenizer.get_weights()},
                 open("image_text_tokeniser.pkl", "wb"))
    else:
        tokenizer_info = pkl.load(open("image_text_tokeniser.pkl", "rb"))
        tokenizer = tf.keras.layers.TextVectorization.from_config(tokenizer_info["config"])
        tokenizer.set_weights(tokenizer_info["weights"])

    return tokenizer


def GetTokenizerVectorAndIndex(tokenizer):
    # Create the tokenized vectors
    tokenizer_vector = caption_dataset.map(lambda x: tokenizer(x))

    # Create mappings for words to indices and indicies to words.
    word_to_index = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
    index_to_word = tf.keras.layers.StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

    return tokenizer_vector, word_to_index, index_to_word


def GetImgCapVector(tock_vector, image_name_vector):
    img_cap_vector = collections.defaultdict(list)
    for img, cap in zip(image_name_vector, tock_vector):
        img_cap_vector[img].append(cap)

    return img_cap_vector


def GetTrainingAndValidationSets(img_cap_vector):
    img_keys = GetShuffledKeys(img_cap_vector)
    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []

    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    return img_name_train, cap_train, img_name_val, cap_val


def GetPredictionDataSet(img_name_train, cap_train, buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(map_func, [item1, item2], [tf.float32, tf.int64]),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == '__main__':
    skip = True

    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1500)

    df = GetImageTextDF()

    if exists('image_path_to_caption.json'):
        train_captions, img_name_vector = GetTrainCaptions()
    else:
        train_captions, img_name_vector = GetTrainCaptions(df=df)

    image_features_extract_model = GetInitializedImageModel()

    if not skip:
        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

        for img, path in image_dataset:
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())

    train_captions = [str(element) for element in train_captions]
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    # Max word count for a caption.
    max_length = 50
    # Use the top 5000 words for a vocabulary.
    vocabulary_size = 5000

    if not exists("image_text_tokeniser.pkl"):
        tokenizer = GetImageTextTokenizer(caption_dataset=caption_dataset, max_caption_length=max_length,
                                          vocab_size=vocabulary_size)
    else:
        tokenizer = GetImageTextTokenizer()

    tokenizer_vector, word_to_index, index_to_word = GetTokenizerVectorAndIndex(tokenizer=tokenizer)
    img_to_cap_vector = GetImgCapVector(tock_vector=tokenizer_vector, image_name_vector=img_name_vector)
    img_name_train, cap_train, img_name_val, cap_val = GetTrainingAndValidationSets(img_cap_vector=img_to_cap_vector)
    cap_train = [numpy.resize(element, 50) for element in cap_train]
    # Feel free to change these parameters according to your system's configuration

    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    prediction_dataset = GetPredictionDataSet(img_name_train=img_name_train,cap_train=cap_train,buffer_size=BUFFER_SIZE,
                                              batch_size=BATCH_SIZE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,decoder=decoder,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)


    #trainning
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    EPOCHS = 2

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(prediction_dataset):
            batch_loss, t_loss = train_step(img_tensor, target)
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                print(f'Epoch {epoch + 1} Batch {batch} Loss {average_batch_loss:.4f}')
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        ckpt_manager.save()

        print(f'Epoch {epoch + 1} Loss {total_loss / num_steps:.6f}')
        print(f'Time taken for 1 epoch {time.time() - start:.2f} sec\n')


    #GenerateMeme
    template_id_df = pd.read_excel("templates.xlsx", dtype={"template": str, "id": int})

    template_row_id = RandomTemplateID(template_id_df.shape[0])
    template_data = template_id_df.iloc[[template_row_id]]
    template_id = template_data["id"].values[0]
    template_name = template_data["template"].values[0]
    templates_img_path = "memes_original/img/{img_name}.jpg".format(img_name=template_name)

    result, attention_plot = evaluate(templates_img_path)

    SendMemeToImgFlip(template_id, result)
