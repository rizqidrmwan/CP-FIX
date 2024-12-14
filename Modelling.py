#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import matplotlib.pyplot as plt
import string


# In[169]:


df = pd.read_csv('data.csv')


# In[170]:


df.head(50)


# In[171]:


#Variabel baru
df_copy = df.copy()


# In[172]:


Sentiment = []
for index, row in df_copy.iterrows():
    if row['score'] == 1 or row['score'] == 2:
        Sentiment.append('Negatif')
    elif row['score'] == 3:
        Sentiment.append('Netral')
    else:  # row['score'] == 4 atau 5
        Sentiment.append('Positif')

df_copy['Sentiment'] = Sentiment
df_copy.to_csv('Data_Sentimented.csv', index=False)
df_copy.head(50)


# PREROCESSING

# In[173]:


#ubah data content menjadi huruf kecil
def casefolding(Review):
    Review = Review.lower()
    return Review
df_copy['content'] = df_copy['content'].apply(casefolding)
df_copy.head(50)


# In[174]:


#normalisasi kata ex:ttp(tetap)
norm = {'ttp' : 'tetap'}

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text   

df_copy['content'] = df_copy['content'].apply(lambda x: normalisasi(x))
df_copy.head(50)


# In[175]:


#Stopwords(menghilangkan kata yang tidak punya makna yang cukup)
more_stop_words =[]

stop_words = StopWordRemoverFactory().get_stop_words()
new_array = ArrayDictionary(stop_words)
stop_words_remover_new = StopWordRemover(new_array)

def stopword(str_text):
    str_text = stop_words_remover_new.remove(str_text)
    return str_text
df_copy['content'] = df_copy['content'].apply(lambda x: stopword(x))
df_copy.head(50)


# In[176]:


#Tokenize(memisahkan kalimat menjadi kata)
tokenized = df_copy['content'].apply(lambda x:x.split())
tokenized


# In[177]:


#Stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
def stemming(content):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in content:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean = []
    d_clean = " ".join(do)
    print(d_clean)
    return d_clean

tokenized = tokenized.apply(stemming)  
tokenized.to_csv('databersih2.csv', index=False)
data_clean = pd.read_csv('Data.csv', encoding='latin1')
data_clean .head(50)


# In[178]:


#Gabungkan 2 atribut
at1 = pd.read_csv('databersih2.csv')
at2 = pd.read_csv('Data_Sentimented.csv')
att2 = at2[['score', 'thumbsUpCount', 'at', 'userName','Sentiment']]

result = pd.concat([at1,att2], axis=1)
# Menyimpan hasil penggabungan ke file baru
result.to_csv('DataFix.csv', index=False)
result.head(50)


# In[179]:


# memeriksa apakah ada teks yang hilang atau null dalam data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
content = result['content']
content.isnull().sum()


# In[180]:


content = content.fillna('tidak ada komentar')


# In[181]:


# convert to numeric
cv = CountVectorizer()
tern_fit = cv.fit(content)

print(len(tern_fit.vocabulary_))


# In[182]:


tern_fit.vocabulary_


# In[183]:


tern_frequency_all = tern_fit.transform(content)
print(tern_frequency_all)


# In[184]:


# menyaring dan memproses data(Sentiment tertentu) & menangani nilai yang hilang
train_s0 = df_copy[df_copy['Sentiment'] == 'Negatif']
train_s0['content'] = train_s0['content'].fillna('tidak ada komentar')
train_s0.head(50)


# In[185]:


# membuat dan menampilkan visualisasi Word Cloud(kata yang sering muncul)
from wordcloud import WordCloud
all_text_s0 = ' '.join(word for word in train_s0['content'])
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s0)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('ulasan negatif')
plt.margins(x=0, y=0)
plt.show()


# In[186]:


# menyaring dan memproses data(Sentiment tertentu) & menangani nilai yang hilang
train_s1 = df_copy[df_copy['Sentiment'] == 'Netral']
train_s1['content'] = train_s1['content'].fillna('tidak ada komentar')
train_s1.head(50)


# In[187]:


# membuat dan menampilkan visualisasi Word Cloud(kata yang sering muncul)
all_text_s1 = ' '.join(word for word in train_s0['content'])
wordcloud = WordCloud(colormap='Greens', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s1)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('ulasan Netral')
plt.margins(x=0, y=0)
plt.show()


# In[188]:


# menyaring dan memproses data(Sentiment tertentu) & menangani nilai yang hilang
train_s2 = df_copy[df_copy['Sentiment'] == 'Positif']
train_s2['content'] = train_s2['content'].fillna('tidak ada komentar')
train_s2.head(50)


# In[189]:


# membuat dan menampilkan visualisasi Word Cloud(kata yang sering muncul)
all_text_s2 = ' '.join(word for word in train_s0['content'])
wordcloud = WordCloud(colormap='Blues', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text_s2)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('ulasan positif')
plt.margins(x=0, y=0)
plt.show()


# In[190]:


# Pie chart
sentimen_data = pd.value_counts(df_copy['Sentiment'], sort=True)
sentimen_data.plot(kind='pie', colors=['Red', 'Green', 'Blue'])
plt.title('Pie Chart')
plt.show


# In[191]:


# menganalisis dan memvisualisasikan distribusi Sentiment sentimen dalam data
sentimen_data = pd.value_counts(df_copy['Sentiment'], sort=True)
sentimen_data.plot(kind='bar', color=['Red', 'Green', 'Blue'])
plt.title('Bar Chart')
plt.show


# SPLIT DATA(TF-IDF)

# In[192]:


# membagi dataset train & test 80% untuk training dan 20% untuk testing
from sklearn.model_selection import train_test_split
result['content'] = result['content'].fillna('Tidak ada komentar')
x_train, x_test, y_train, y_test = train_test_split(result['content'], result['Sentiment'], test_size=0.2, stratify=result['Sentiment'], random_state=30)


# In[193]:


# Hitung jumlah data latih dan data uji
train_size = len(x_train)
test_size = len(x_test)


# In[194]:


# Hitung persentase
train_percentage = (train_size / (train_size + test_size)) * 100
test_percentage = (test_size / (train_size + test_size)) * 100
print(f"Data latih: {train_percentage:.2f}%")
print(f"Data uji: {test_percentage:.2f}%")


# In[195]:


import numpy as np


# In[196]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

print(x_train.shape)
print(x_test.shape)


# In[197]:


x_train = x_train.toarray()
x_test = x_test.toarray()


# MECHINE LEARNING(NLP MENGGUNAKAN ALGORITMA NAIVE BAYES)
# 

# In[198]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()


# In[199]:


# melakukan hyperparameter tuning
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

#Teknik cross-validation
cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=999)

params_NB = {'var_smoothing': np.logspace(0, -9, num=100)}
#pelatihan model menggunakan GridSearchCV
gscv_nb = GridSearchCV(estimator=nb, param_grid=params_NB, cv= cv_method, verbose=1, scoring= 'accuracy')
gscv_nb.fit(x_train, y_train)
gscv_nb.best_params_


# In[200]:


# membuat sebuah model Naive Bayes berbasis Gaussian dengan parameter var_smoothing
nb = GaussianNB(var_smoothing=0.001)


# LATIH MODEL DATA TRAINING

# In[201]:


nb.fit(x_train, y_train)


# PREDIKSI DATA PADA DATA TESTING

# In[202]:


y_pred_nb = nb.predict(x_test)
print(y_pred_nb)


# EVALUASI (CONFUSION MATRIX)

# In[203]:


from sklearn.metrics import RocCurveDisplay, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[204]:


print("train Accuracy : ",nb.score(x_train,y_train))
print("test Accuracy : ",nb.score(x_test,y_test))


# In[205]:


print('----- confusion matrix -----')
print(confusion_matrix(y_test, y_pred_nb))

print('----- classification report -----')
print(classification_report(y_test, y_pred_nb))

# Evaluasi: Akurasi dalam bentuk persen
accuracy = accuracy_score(y_test, y_pred_nb) * 100

print(f"\nAkurasi Model: {accuracy:.2f}%")

