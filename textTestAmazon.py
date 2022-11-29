# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
import re, string, nltk
from wordcloud import WordCloud, STOPWORDS
import emoji
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv("./data/AmazonReview/train.csv",header=None)
df.columns=['sentiment','title','text']
df = df.dropna()
df1 = df.sample(frac=0.05)
# df1.sentiment.value_counts()
# sns.countplot(df1.sentiment,palette="mako")
# plt.title("Countplot for Sentiment Labels")
# plt.show()
df1["title"] = df1["title"].astype(str)
def clean_text(df, field):
    df[field] = df[field].str.replace(r"@"," at ")
    df[field] = df[field].str.replace("#[^a-zA-Z0-9_]+"," ")
    df[field] = df[field].str.replace(r"[^a-zA-Z(),\"'\n_]"," ")
    df[field] = df[field].str.replace(r"http\S+","")
    df[field] = df[field].str.lower()
    return df

df1 = clean_text(df1,"title")
lemmatizer = WordNetLemmatizer()
#stemmer = SnowballStemmer("english")
import re
#Removes Punctuations
def remove_punctuations(data):
    punct_tag=re.compile(r'[^\w\s]')
    data=punct_tag.sub(r'',data)
    return data

#Removes HTML syntaxes
def remove_html(data):
    html_tag=re.compile(r'<.*?>')
    data=html_tag.sub(r'',data)
    return data

#Removes URL data
def remove_url(data):
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data

#Removes Emojis
def remove_emoji(data):
    emoji_clean= re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0" 
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    data=emoji_clean.sub(r'',data)
    url_clean= re.compile(r"https://\S+|www\.\S+")
    data=url_clean.sub(r'',data)
    return data

#Lemmatize the corpus
def lemmatize_text(data):
   # lemmatizer=WordNetLemmatizer()
    out_data=""
    for words in data:
        out_data+= lemmatizer.lemmatize(words)
    return out_data

#df1["title_clean"] = df1["title"].apply(preprocess_text)
df1["title_clean"] = df1["title"].apply(lambda x: remove_punctuations(x))
df1["title_clean"] = df1["title_clean"].apply(lambda x: remove_emoji(x))
df1["title_clean"] = df1["title_clean"].apply(lambda x: remove_html(x))
df1["title_clean"] = df1["title_clean"].apply(lambda x: remove_url(x))
df1["title_clean"] = df1["title_clean"].apply(lambda x: lemmatize_text(x))
df1.sentiment.replace({1:0,2:1},inplace=True)
df = df1[["sentiment","title_clean"]]
from sklearn.model_selection import train_test_split
X = df["title_clean"]
y = df.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(X_train)
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("spacy", language="en")
X_train_seq = X_train.apply(word_tokenize)
X_test_seq = X_test.apply(word_tokenize)
print()



import torch.nn
bilstm = torch.nn.LSTM(10, 20, 2)