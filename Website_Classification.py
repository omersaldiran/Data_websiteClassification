#!/usr/bin/env python
# coding: utf-8
# First, import necessary lib which is NLTK and download if don't have

# In[1]:


import nltk
nltk.download('stopwords')


# İmport necessary libraries

# In[2]:


from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.corpus import stopwords, reuters
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
stop_words = stopwords.words("english")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Reading CSV file which is downloaded from Kaggle website. I'm using website classification dataset.
# https://www.kaggle.com/hetulmehta/website-classification

# In[3]:


data = pd.read_csv("website_classification.csv")


# Checking the first 10 entry of dataset

# In[4]:


data.head(10)


# Checing the each category of the dataset

# In[6]:


data['Category'].value_counts()


# Defining the vectorizer model which is Count Vectorizer

# In[7]:


vectorizer = CountVectorizer(analyzer = "word", max_features = 10, max_df=0.3)
count_model = vectorizer.fit(data["cleaned_website_text"])
X = count_model.transform(data["cleaned_website_text"])


# Features that extracted from dataset

# In[8]:


count_model.get_feature_names()


# In[9]:


X.todense()[:5]


# In[10]:


X.shape


# Preparing full model for the prediction of category of the websites.
# Tfid vectorizer is used for metric.
# I've used a big dataset before this dataset which is "ireland-news-headlines.csv" but it has too many different categories. Classification process took too much time and the confusion matrix was not clear.
# After that I've changed the dataset and used website_classification.
# 
# Classification result is looking good if we check the classification report.

# In[11]:


vectorizer = TfidfVectorizer(analyzer = "word", max_features = 1000)
tfidf_model = vectorizer.fit(data["cleaned_website_text"])
pickle.dump(tfidf_model, open("tfidf.pkl", "wb"))
X = tfidf_model.transform(data["cleaned_website_text"])
X_train,X_test,y_train,y_test = train_test_split(X,data["Category"],test_size = 0.1)
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)
pickle.dump(clf, open("text_clf.pkl", 'wb'))
preds = clf.predict(X_test)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))


# In[ ]:




