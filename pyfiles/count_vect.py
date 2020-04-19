from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import scipy.sparse

newsA = pd.read_csv('data/articles1.csv')
newsB = pd.read_csv('data/articles2.csv')
newsC = pd.read_csv('data/articles3.csv')

corpusA = newsA.content
corpusB = newsB.content
corpusC = newsC.content

P1 = pd.concat([corpusA, corpusB])
corpus = pd.concat([P1, corpusC])
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
feat = vectorizer.get_feature_names()
np.save('vect_feat.npy', feat)
