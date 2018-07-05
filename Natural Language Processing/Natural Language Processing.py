# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 12:33:30 2018

@author: Kaustub Sinha
"""

#import pandas as pd
#
## Importing the dataset
#dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#
## Cleaning the texts
#import re
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#corpus = []
#for i in range(0, 1000):
#    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
#    review = review.lower()
#    review = review.split()
#    ps = PorterStemmer()
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review = ' '.join(review)
#    corpus.append(review)
#
## Creating the Bag of Words model
#from sklearn.feature_extraction.text import CountVectorizer
#cv = CountVectorizer(max_features = 1500)
#features = cv.fit_transform(corpus).toarray()
#labels = dataset.iloc[:, 1].values
#
## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)
#
## Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(features_train, labels_train)
#
## Predicting the Test set results
#labels_pred = classifier.predict(features_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(labels_test, labels_pred)

'''
Code Challenge

'''
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('movie.csv')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 2000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
features = cv.fit_transform(corpus).toarray()
labels = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(features_train, labels_train)

# Predicting the Test set results
labels_pred = classifier.predict(features_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, labels_pred)

corpus1=[]
x =  " plot   odin is a great high school basketball player    he s dating a hot girl and the coach loves his ass    in fact   the coach even admits to having fatherly feelings towards him    unfortunately   the coach s real son   hugo   isn t too pleased to hear that    in fact   he doesn t like hearing about any of odin s triumphs   as they generally supersede his own    so what does he set out to do    well   let s just say that he starts to mess with people s heads and one thing leads to another thing which leads to       well    you ll see    critique   a very powerful   thoroughly depressing   well acted   non teen   movie starring a bunch of teens    credit director tim blake nelson for creating a modern day version of this shakespearian classic   set in a realistic high school environment   with the basketball championships as the backdrop and an impending sense of doom as it core    you also have to give it up to all of the actors in this film   who turn over some very convincing performances   taking you through all of their characters  ups and downs    hartnett especially should be applauded for taking on such a despicable character   a dude who you just want to grab by the throat and beat the shit out of   sign of a good actor   if you ask me      phifer also comes to play in this movie   with a nice blend of charisma   fear   love and anger spread over his character    i was always on this guy s side and definitely felt sorry for him as things moved along    julia stiles was also good   but her character wasn t much different from others that she s played recently    i was however very surprised with martin sheen s showing   since i hadn t taken him too seriously as an actor over the past few years    his character is definitely over the top in this film   but i appreciated his fervor   his rage   his ultimate and blind desire to win above all    a great example of a workaholic man who cannot see the trees from the forest    i did have a few reservations about the movie though    first of all   what was with all of the gangsta hip hop music used in pretty much every other scene transition    it was cute at first   but became a little too obvious and annoying after a few times    the film was also edited pretty choppily       like  some scenes were cut out and no one   cleaned it up   afterwards    i also didn t like the fact that hartnett s character   the man behind much of the nastiness that goes down in this film   was infallible    in other words   pretty much everything he says or asks of someone   happens automatically and without any goofups    i mean   i know the guy is smart and all   but i would have appreciated a little more   realism   under some of the circumstances    but overall   the movie will devastate you    it s not a   fun date movie      it s not a movie about the high school basketball team and how its black star falls for the white girl from the other side of the tracks    it s about jealousy   love   envy   fury   passion   revenge and pretty much any other negative thought that s ever passed through your head    is it worth seeing    oh   most definitely !     pun intended   it s a moving picture with great performances   no shakespearian speak   thank you   god !   and an outstanding directorial job by nelson    the final scene alone is enough to send a massive chill down your spine   and i was especially impressed with the director s choice of music near the end    in fact   the whole sequence was quite reminiscent of the final showdown scene from taxi driver   and i guess that s saying something right there    where s joblo coming from    10 things i hate about you   7/10     election   7/10     hamlet   6/10     love s labor s lost   8/10     natural born killers   9/10     save the last dance   7/10     shakespeare in love   5/10    " 
#x = "Its awesome to have Class Forsk"
review1 = re.sub('[^a-zA-Z]', ' ', x)
review1 = review1.lower()
review1 = review1.split()
ps = PorterStemmer()
review1 = [ps.stem(word) for word in review1 if not word in set(stopwords.words('english'))]
review1 = ' '.join(review1)
corpus1.append(review1)

features_new = cv.transform(corpus1).toarray()

New_Str_prid = classifier.predict(features_new)




from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2)
classifier.fit(features_train,labels_train)

Pred = classifier.predict(features_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,Pred)

Score = classifier.score(features_test,labels_test)