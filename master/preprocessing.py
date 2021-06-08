import tensorflow as tf
from reading_file import read_img_caption

#reading images and captions and ids
imgs , captions ,ids = read_img_caption()


def preprocess_captions_vector(captions=captions):

    num_words = 5000

    #calculating the maximum of tokens in all captions
    maximum_token = max(len(caption.split(" ")) for caption in captions)

    #tokenizing captions
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words,filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
                                                      oov_token='<unk>')
    #making a map for any words
    tokenizer.fit_on_texts(captions)

    #adding a <pad> in map , word_index is your map
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    #convetring text to vector
    captions_seq = tokenizer.texts_to_sequences(captions)

    captions_vector = tf.keras.preprocessing.sequence.pad_sequences(captions_seq,padding='post').tolist()

    return captions_vector,maximum_token


def preprocess_img(imgs=imgs):

    images = tf.keras.applications.inception_v3.preprocess_input(imgs)

    return images

def split_data(caption_vector,images):

    slice = int(len(caption_vector)*0.8)

    caption_vector_train = caption_vector[0:slice]
    caption_vector_val = caption_vector[slice:]
    images_train = images[0:slice]
    images_val = images[slice:]

    return caption_vector_train,caption_vector_val,images_train,images_val

def tf_dataset(images,caption_vector):

    dataset = tf.data.Dataset.from_tensor_slices((images,caption_vector))

    return dataset