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

### Load news dataset 
news = pd.read_csv('data/articles.csv')
### Specify which rows to include. Each row contains one article. More rows requires longer runtime. 
news = news.iloc[:500, :]
### This step combines the article headline & content in one string cell. 
news['full_text'] = news.title.str.cat(news.content, sep='.')

def row_counter(row):
    ### Set variable to equal the string in column 'full_text'
    text = news.full_text.iloc[row]
    ### Separate strings into distinct sentence fragments by splitting each string on punctuation 
    text = re.split('[?.,—!;:\’\'"“”\(\)]', text)
    ### Convert each sentence fragment string to a list of words by splitting on spaces
    frags = [x.split(' ') for x in text]
    ### Filter out empty strings 
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
    phrase_df['combo'] = phrase_df.w1.str.cat(phrase_df.w2.values)
    phrase_df = phrase_df[phrase_df.combo.str.isalpha()]

    ### Concatenate the first word with the second word.
    phrase_df['combo'] = phrase_df.w1.str.cat(phrase_df.w2, sep=' ')

    ### Filter out phrases that contain capital letters. 
    phrase_df = phrase_df[phrase_df.combo.str.islower()]
    
    stop_words = set(stopwords.words('english'))

    phrase_df = phrase_df[~phrase_df.w1.isin(stop_words)]
    phrase_df = phrase_df[~phrase_df.w2.isin(stop_words)]

    phrase_df['lem1'] = phrase_df.w1.apply(lambda x: lemmatizer.lemmatize(x))
    phrase_df['lem2'] = phrase_df.w2.apply(lambda x: lemmatizer.lemmatize(x)) 

    common = ['year', 'month', 'week', 'day', 'million', 
              'thousand', 'could', 'said', 'told', 'would', 'one', 
              'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
              'nine']
    
    phrase_df = phrase_df[~phrase_df.lem1.isin(common)]
    phrase_df = phrase_df[~phrase_df.lem2.isin(common)]

    phrase_df['phrase'] = phrase_df.lem1.str.cat(phrase_df.lem2, sep=' ')

    ### Choose where to save the dataframe
    phrase_df.to_csv('art3B.csv')

    phrases = phrase_df.phrase.values
    c = Counter(phrases)
    return c

big_counter = Counter()
for row in range(len(news)):
    # if row % 50 == 0:
    if row % 1000 == 0:
        print(row)
    rc = row_counter(row)
    big_counter += rc



with open('art3B.pickle', 'wb') as outputfile:
    pickle.dump(big_counter, outputfile)