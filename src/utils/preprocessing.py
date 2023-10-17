import re
import string
import nltk
from nltk import word_tokenize


def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)


def preprocess_text(text):
    """
    preprocess one string (text):
    1. remove URLs
    4. remove emojis # to description
    5. to lower case
    7. remove non-ascii

    """
    # remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',
                '', text)
    text = re.sub(r'http\S+', '', text)
    # remove emojis
    text = remove_emojis(text)
    # convert text to lower-case
    text = text.lower()
    # remove non-ascii characters
    #text = re.sub(r'[^a-zA-Z]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def preprocess(corpus):
    """
    preprocess a list of strings (texts)
    """
    outcorpus = []
    for text in corpus:
        outcorpus.append(preprocess_text(text))
    return outcorpus
