import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models import FastText

import joblib

contract_model = FastText.load("contracts_fasttext.model")

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=1000,
                      verbose=5, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    joblib.dump(tsne_model, "tsne_model.scikit")
    joblib.dump(new_values, "new_values.scikit")

    #x = []
    #y = []
    #for value in new_values:
    #    x.append(value[0])
    #    y.append(value[1])
    #    
    #f = plt.figure(figsize=(16, 16)) 
    #for i in range(len(x)):
    #    plt.scatter(x[i],y[i])
    #    plt.annotate(labels[i],
    #                 xy=(x[i], y[i]),
    #                 xytext=(5, 2),
    #                 textcoords='offset points',
    #                 ha='right',
    #                 va='bottom')
    #f.savefig("fasttext_f.pdf",bbox_inches='tight')
    #plt.savefig("fasttext_plt.pdf",bbox_inches='tight')
    #f.savefig("fasttext_f.png",bbox_inches='tight')
    #plt.savefig("fasttext_plt.png",bbox_inches='tight')
    #plt.show()

tsne_plot(contract_model)