import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from collections import Counter
import pickle

lemmatizer = WordNetLemmatizer()

news = pd.read_csv('data/articles1.csv')
news = news.iloc[:1000, :]
news['full_text'] = news.title.str.cat(news.content, sep='.')

def row_counter(row):
    text = news.full_text.iloc[row]
    text = re.split('[?.,—!;:\’\'"“”\(\)]', text)
    frags = [x.split(' ') for x in text]
    for x in frags:
        while x.count('') > 0:
            x.remove('')
    
    word1_list = []
    word2_list = []
    for each_frag in frags:
        if len(each_frag) > 1:
            for idx in range(len(each_frag)-1):
                word1_list.append(each_frag[idx])
                word2_list.append(each_frag[idx+1])
    
    arr = np.array([word1_list, word2_list]).T
    phrase_df = pd.DataFrame(data=arr, columns=['w1', 'w2'])
    phrase_df['combo'] = phrase_df.w1.str.cat(phrase_df.w2.values, sep=' ')
    phrase_df = phrase_df[phrase_df.combo.str.isalpha()]
    phrase_df = phrase_df[phrase_df.combo.str.islower()]

    # phrase_df['lower1'] = phrase_df.w1.str.lower()
    # phrase_df['lower2'] = phrase_df.w2.str.lower()
    
    stop_words = set(stopwords.words('english'))

    phrase_df = phrase_df[~phrase_df.w1.isin(stop_words)]
    phrase_df = phrase_df[~phrase_df.w2.isin(stop_words)]

    ### not sure whether to feed in w1/w2 OR lower1/lower2
    # phrase_df['lem1'] = phrase_df.lower1.apply(lambda x: lemmatizer.lemmatize(x))
    # phrase_df['lem2'] = phrase_df.lower2.apply(lambda x: lemmatizer.lemmatize(x))

    # phrase_df['phrase'] = phrase_df.lem1.str.cat(phrase_df.lem2, sep=' ')
    
    phrases = phrase_df.combo.values
    c = Counter(phrases)
    return c

big_counter = Counter()
for row in range(len(news)):
    # if row % 500 == 0:
    if row % 100 == 0:
        print(row)
    rc = row_counter(row)
    big_counter += rc



with open('pk_test.pickle', 'wb') as outputfile:
    pickle.dump(big_counter, outputfile)
