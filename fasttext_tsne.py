# This is just code we'll need later on, allowing us to generate a t-SNE plot 
# (the "standard", though perhaps not best, way to visualize an embedding space)
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
def tsne_plot(model):
    labels = []
    tokens = []
    for word in model.wv.vocab:
        tokens.append(model.wv[word])
        labels.append(word)
    tsne_model = TSNE(perplexity=40, n_components=2, init="pca", n_iter=2000, random_state=1948)
    new_values = tsne_model.fit_transform(tokens)
    my_x = new_values[:,0]
    my_y = new_values[:,1]
    plt.figure(figsize=(32, 32))
    a = pd.DataFrame({'x': my_x, 'y': my_y, 'val': labels})
    a_sample = a.sample(1500)
    counter = 0
    for i, point in a_sample.iterrows():
        plt.scatter(point['x'], point['y'])
        plt.text(point['x']+.02, point['y'], str(point['val']), fontsize=8)
        counter = counter + 1
    plt.show()