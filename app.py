import gradio
import keras
import nltk
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import logging
# Set up logging
logging.basicConfig(filename='predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

nltk.download("stopwords")
import re

vocab_size = 25000
embedding_dim = 64
max_length = 50
trunc_type = 'pre'
padding_type = 'pre'
oov_tok = '<OOV>'

labels = {
    1: 'admiration',
    2: 'anger',
    3: 'curiosity',
    4: 'disappointment',
    5: 'disgust',
    6: 'embarrassment',
    7: 'excitement',
    8: 'fear',
    9: 'gratitude',
    10: 'love',
    11: 'nervousness',
    12: 'pride',
    13: 'sadness',
    14: 'neutral'
}


def get_tokenizer():
    df = pd.read_csv('util/temp.csv', ).reset_index(drop=True)
    X = df['text']
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X)
    return tokenizer


def clean_text(text):
    stemmer = PorterStemmer()
    stp_words = set(stopwords.words('english'))
    sent = text.replace("NAME", "")
    sent = re.sub('[^a-zA-Z]', " ", sent)
    sent = sent.lower()
    sent = sent.split()
    sent = [stemmer.stem(words) for words in sent if words not in stp_words]
    text = [' '.join(sent)]
    # print(text)
    # print((sent))

    train_seq = get_tokenizer().texts_to_sequences(text)
    # print(train_seq)
    test_padding = pad_sequences(train_seq, max_length, padding=padding_type, truncating=trunc_type)

    return test_padding


def predict(text):
    model = keras.models.load_model('model/cnn_model.h5')
    pred = model.predict(text)
    pred_class = np.argmax(pred, axis=-1)

    return labels.get(pred_class[0])


def my_inference_function(text):
    """do the pre-processing of text to be able to feed it
    in the model"""
    # 1. Preprocess
    inp = clean_text(text)
    output = predict(inp)

    logging.info(f"Input Text: {text}  Model Prediction: {output}")
    return output


gradio_interface = gradio.Interface(
    fn=my_inference_function,
    inputs="text",
    outputs="text",
    # outputs=gradio.outputs.Label(num_top_classes=5),
    title='Emotion Classification',
    description=f"Let's know how you feel!!!",
    allow_flagging="never"
)

gradio_interface.launch(inbrowser=True)
