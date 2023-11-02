# Importing the libraries needed for Exploratory Data Analysis (EDA)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')
 
# Importing libraries necessary for Model Building and Training
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
 
import warnings
warnings.filterwarnings('ignore')

""" Read data using Pandas """
data = pd.read_csv('spam.csv')
data.head()
# prints the first 5 rows of data
# print(data.head())
# data.shape
# shows the shape of the data
# print(data.shape)

#Plots these counts
sns.countplot(x ='Category', data=data)
plt.title("Distribution of Spam & Ham Email Messages")
plt.xlabel("Message Types")
plt.show()

""" Text Processing """
punctuations_list = string.punctuation

def remove_punct(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

data['Message'] = data['Message'].apply(lambda x: remove_punct(x))
# print(data.head())

# Function to remove stop words
def remove_Stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []

    # Storing important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)
        
    output = " ".join(imp_words)

    return output

data['Message'] = data['Message'].apply(lambda text: remove_Stopwords(text))
print(data.head())

""" Word Cloud Text Visualization """
def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['Message'])
    plt.figure(figsize=(7,7))

    wc = WordCloud(background_color='black', max_words=100, 
                   width=800,
                   height=400, 
                   collocations=False).generate(email_corpus)

    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} Emails', fontsize=15)
    plt.axis('off')
    plt.show()

plot_word_cloud(data[data['Category'] == 'ham'], typ = 'Non-Spam')
plot_word_cloud(data[data['Category'] == 'spam'], typ = 'Spam')

""" Word to Vector conversion with token IDs (machine learning model
 only works on numbers) """
# Train test split
train_X, test_X, train_Y, test_Y = train_test_split(data['Message'],
                                                    data['Category'],
                                                    test_size=0.2,
                                                    random_state=42)

# Tokenize the message data
tokenizer = Tokenizer()

# Convert message to sequences
train_seq = tokenizer.texts_to_sequences(train_X)
test_seq = tokenizer.texts_to_sequences(test_X)

# Pad sequences to have the same length
max_len = 100
train_seq = pad_sequences(train_seq,
                          maxlen=max_len,
                          padding='post',
                          truncating='post')
test_seq = pad_sequences(test_seq,
                         maxlen=max_len,
                         padding='post',
                         truncating='post')

""" Build the model """
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,
                                    output_dim=32,
                                    input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Print the model summary
model.summary()

# Compile the model with these three essential parameters
# Optimizer - method that is used to help optimize the cost
# function by using gradient descent
# Loss - function by which we monitor whether the model is 
# improving with training or not
# Metrics - helps evaluate the model by predicting the training
# and validation data
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'],
              optimizer='adam')

# We use Callbacks to check whether the model is improving with
# each epoch or not
es = EarlyStopping(patience=3,
                   monitor='val_accuracy',
                   restore_best_weights=True)

lr = ReduceLROnPlateau(patience=2,
                       monitor='val_loss',
                       factor=0.5,
                       verbose=0)

# We train the model
labelencoder = LabelEncoder()
train_Y = labelencoder.fit_transform(train_Y) # Convert categorical value to its numerical form
test_Y = labelencoder.fit_transform(test_Y)
history = model.fit(train_seq, train_Y,
                    validation_data=(test_seq, test_Y),
                    epochs=20,
                    batch_size=32,
                    callbacks=[lr, es])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_seq, test_Y)
print('Test Loss :',test_loss)
print('Test accuracy :',test_accuracy)

# Model Evaluation by plotting a graph depicting the variance of training
# and validation accuracies with the number of epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()