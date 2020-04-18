import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer 
from collections import Counter
import pickle

#lemmatizer = WordNetLemmatizer()

news = pd.read_csv('data/articles1.csv')
news = news.iloc[4000:10000, :]
# news['full_text'] = news.title.str.cat(news.content, sep='.')

def row_counter(row):
    text = news.content.iloc[row]
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
    #filter out any numbers or remaining punctuation
    phrase_df = phrase_df[phrase_df.w1.str.isalpha()]
    phrase_df = phrase_df[phrase_df.w2.str.isalpha()]

    phrase_df['combo'] = phrase_df.w1.str.cat(phrase_df.w2.values, sep=' ')

    phrase_df = phrase_df[phrase_df.combo.str.islower()]
    
    stop_words = set(stopwords.words('english'))
    # common = ['year', 'month', 'week', 'day', 'million', 
    #           'thousand', 'could', 'said', 'told', 'would', 'one', 
    #           'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
    #           'nine']
    # for word in common:
    #     stop_words.add(word)

    phrase_df = phrase_df[~phrase_df.w1.isin(stop_words)]
    phrase_df = phrase_df[~phrase_df.w2.isin(stop_words)]

    ### phrase_df = phrase_df[~phrase_df.lower2.isin(stop_words)]
    ### phrase_df = phrase_df[~phrase_df.lower2.isin(stop_words)]

    ### not sure whether to feed in w1/w2 OR lower1/lower2
    # phrase_df['lem1'] = phrase_df.w1.apply(lambda x: lemmatizer.lemmatize(x))
    # phrase_df['lem2'] = phrase_df.w2.apply(lambda x: lemmatizer.lemmatize(x))

    # phrase_df['lem3'] = phrase_df.w1.apply(lambda x: lemmatizer.lemmatize(x))
    # phrase_df['lem4'] = phrase_df.w2.apply(lambda x: lemmatizer.lemmatize(x))    

# '''    common = ['year', 'month', 'week', 'day', 'million', 
#               'thousand', 'could', 'said', 'told', 'would', 'one', 
#               'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 
#               'nine', 'made', 'make', 'making', 'liking', 'looking', 'like', 'liked', 'likes', 
#              'many', 'even', 'though', 'look', 'looks', 'looked', 'take', 'took', 'know', 
#              'knew', 'knows', 'say', 'said', 'go', 'got', 'much', 'often', 'who', 'whom', 
#              'whose', 'their', 'theirs', 'dozen', 'might', 'may', 'never', 'also', 'still',
#              'expressed', 'went', 'expresses', 'express', 'expressing', 'saying', 'u']

    
#     phrase_df = phrase_df[~phrase_df.w1.isin(common)]
#     phrase_df = phrase_df[~phrase_df.w2.isin(common)]'''

    # phrase_df['phrase'] = phrase_df.w1.str.cat(phrase_df.w2, sep=' ')

    # phrase_df = phrase_df[phrase_df.phrase.str.islower()]

    phrases = phrase_df.combo.values
    c = Counter(phrases)
    return c

big_counter = Counter()
for row in range(len(news)):
    # if row % 50 == 0:
    if row % 500 == 0:
        print(row)
    rc = row_counter(row)
    big_counter += rc



with open('count1B.pickle', 'wb') as outputfile:
    pickle.dump(big_counter, outputfile)
