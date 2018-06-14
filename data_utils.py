import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import re
import collections
import pickle


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = text.strip().lower()

    return text


def build_dict(train_tsv, is_train=True):
    if is_train:
        df = pd.read_csv(train_tsv, sep="\t")
        sentences = df["sentence"]

        words = list()
        for sentence in sentences:
            for word in word_tokenize(clean_str(sentence)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
    document_max_len = 20

    return word_dict, reversed_dict, document_max_len


def build_dataset(tsv, word_dict, document_max_len):
    df = pd.read_csv(tsv, sep="\t")

    x = list(map(lambda d: word_tokenize(clean_str(d)), df["sentence"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<padding>"]], x))

    y = list(df["sentiment"])

    return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
