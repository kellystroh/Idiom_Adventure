import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
import pickle

df = pd.read_csv('data/articles1.csv')
df = df.iloc[:10000, :]
df.content = df.content.str.split('[?.,—!;:\’\'"“”\(\)]')

df = df.content.apply(lambda x: pd.Series(x)).join(df).melt(['publication', 'id'], var_name='fragment')
df.value = df.value.str.strip().str.split(' ')
df = df[~df.value.isna()]
df = df[df.value.apply(lambda x: len(x)) > 1]
print('step one')

fragments = df.value.tolist()
word1_list = []
word2_list = []
for each_frag in fragments:
    if len(each_frag) > 1:
        for idx in range(len(each_frag)-1):
            word1_list.append(each_frag[idx])
            word2_list.append(each_frag[idx+1])

arr = np.array([word1_list, word2_list]).T
phrase_df = pd.DataFrame(data=arr, columns=['w1', 'w2'])
print('step two')

phrase_df = phrase_df[phrase_df.w1.str.isalpha()]
phrase_df = phrase_df[phrase_df.w2.str.isalpha()]
print('step three')
phrase_df['combo'] = phrase_df.w1.str.cat(phrase_df.w2.values, sep=' ')

phrase_df = phrase_df[phrase_df.combo.str.islower()]
print('step four')
stop_words = set(stopwords.words('english'))
phrase_df = phrase_df[~phrase_df.w1.isin(stop_words)]
phrase_df = phrase_df[~phrase_df.w2.isin(stop_words)]

print('step five')
phrases = phrase_df.combo.values
count = Counter(phrases)


with open('counter_Z.pickle', 'wb') as outputfile:
    pickle.dump(count, outputfile)