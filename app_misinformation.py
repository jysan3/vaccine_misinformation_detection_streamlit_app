import pandas as pd
import numpy as np
import re
import nltk
nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import streamlit as st
import pickle

dt_clf= pickle.load(open("app/decision_tree.pkl", 'rb'))
tfid1 = pickle.load(open("app/tfidf1.pkl", 'rb'))
tfid2 = pickle.load(open("app/tfidf2.pkl", 'rb'))
tfid3 = pickle.load(open("app/tfidf3.pkl", 'rb'))

st.set_page_config(
    page_title='Vaccine misinformation detection App',
    page_icon=':chart_with_upwards_trend:',
    layout="wide")

st.title('Misinformation detection')

@st.cache
# Fonction to predict if a tweet is misinformation
def misinformation(input):
    predd = pd.DataFrame()
    predd['Texts'] = ""
    predd['hashtags'] = ""
    predd['mentions'] = ""
    predd = predd.append({'Texts': input}, ignore_index=True)

    predd['hashtags'] = predd['Texts'].apply(lambda x: re.findall(r'#([a-zA-Z0-9_]{1,50})', x))
    predd['Texts'] = predd['Texts'].apply(lambda x: re.sub("#[A-Za-z0-9_]+", "", x))

    predd['mentions'] = predd['Texts'].apply(lambda x: re.findall('@([a-zA-Z0-9_]{1,50})', x))
    predd['Texts'] = predd['Texts'].apply(lambda x: re.sub("@([a-zA-Z0-9_]{1,50})", "", x))

    def processTweet(tweets):
        tweets = tweets.str.lower()
        tweets = tweets.str.replace(r"<[^<>]+>", " ")
        tweets = tweets.str.replace(r"[^\s]+@[^\s]+", '')
        tweets = tweets.str.replace(r"(http|https)://[^\s]*", '')
        tweets = tweets.str.replace(r"[0-9]+", '')
        tweets = tweets.str.replace(r"[$]+", '')
        transl_table = dict([(ord(x), ord(y)) for x, y in zip(u"‘’´“”–-", u"'''\"\"--")])
        tweets = tweets.apply(lambda a: a.translate(transl_table))
        tweets = tweets.str.replace(r"[^\w]+", ' ')
        return tweets

    predd['Texts'] = processTweet(predd['Texts'])

    stop_words = stopwords.words('english')
    predd['Texts'] = predd['Texts'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    def stem_tweets(tweets):
        stemmer = PorterStemmer()
        tweets = tweets.apply(lambda a: list(map(stemmer.stem, a.split())))
        return tweets

    predd['Texts'] = stem_tweets(predd['Texts'])

    predd['Texts'] = predd['Texts'].apply(lambda x: ' '.join(x))
    predd['hashtags'] = predd['hashtags'].apply(lambda x: ' '.join(x))
    predd['mentions'] = predd['mentions'].apply(lambda x: ' '.join(x))

    df_tfid_text = tfid1.transform(predd['Texts'])
    df_tfid_hashtags = tfid2.transform(predd['hashtags'])
    df_tfid_mentions = tfid3.transform(predd['mentions'])

    X_predd = np.hstack([df_tfid_text.toarray(), df_tfid_hashtags.toarray(), df_tfid_mentions.toarray()])

    prediction = dt_clf.predict(X_predd)
    return prediction

# ------- Text Input -------

news_text = st.text_input("Enter tweet you want to test","Type Here", max_chars=140)
if st.button("Predict"):
    output_info = misinformation(news_text)
    if output_info[0] == 1:
        st.write("This is misinformation")
    else:
        st.write("This is not misinformation")