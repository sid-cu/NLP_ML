
# coding: utf-8

# In[3]:


sentence = "yoyo honey singh in the hoouse for one more shot, who don't wanna miss your's opp"


# In[4]:


sentence.split()


# In[5]:


import nltk


# In[6]:


nltk.download()


# In[7]:


nltk.download('averaged_perceptron_tagger')


# In[13]:


nltk.pos_tag('Machine learning is great'.split())


# In[2]:


import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


# In[3]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[11]:


stopwords = set(w.rstrip() for w in open('/Users/owner/Documents/NLP/Sentiment_Analysis/stopwords.txt'))


# In[12]:


stopwords


# In[27]:


positive_reviews = BeautifulSoup(open('/Users/owner/Documents/NLP/Sentiment_Analysis/electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')


negative_reviews = BeautifulSoup(open('/Users/owner/Documents/NLP/Sentiment_Analysis/electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')


# In[28]:


np.random.shuffle(positive_reviews)

positive_reviews = positive_reviews[:len(negative_reviews)]


# In[40]:


def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>2]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if t not in stop_words]
    return tokens
    


# In[36]:


nltk.download('wordnet')


# In[41]:


word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1
            
            
for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index +=1
            
            
            
            
            


# In[44]:


def token_to_vector(tokens, label):
    x = np.zeros(len(word_index_map)+1)
    for t in tokens:
        i = word_index_map[t]
        x[i] +=1
    
    x = x/x.sum()
    
    x[-1] =label
    return x


# In[45]:


N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map)+1))

i = 0
for tokens in positive_tokenized:
    xy = token_to_vector(tokens,1)
    data[i,:] = xy
    i =+1
    
for tokens in negative_tokenized:
    xy = token_to_vector(tokens,1)
    data[i,:] = xy
    i +=1


# In[46]:


np.random.shuffle(data)
X = data[:,:-1]
Y = data[:,-1]


# In[47]:


X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]


# In[49]:


from sklearn.linear_model import LogisticRegression


# In[50]:


model = LogisticRegression()
model.fit(X_train, Y_train)
print("Train accuracy:", model.score(X_train, Y_train))
print("Test accuracy:", model.score(X_test, Y_test))

