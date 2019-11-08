import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models import FastText

import joblib

contract_model = FastText.load("contracts_fasttext.model")

from fasttext_tsne import tsne_plot

tsne_plot(contract_model)