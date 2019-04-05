from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model

"""
Task : In this exercise we try to generate some text using a text corpus (provided as input text file).
Please follow the below code and use it to write a function which, given a seed word will generate couple of
words by repeatedly calling model. For example, if you supply seed word = "the", below model outputs "driver".
Now, you can further call the model with seed word = "driver" and generate a sentence.
You are free to modify the model as you seem fit and also you can choose a large text corpus.

Note : The seed word sould be from vocabulary of text corpus, this means that model only knows about
       the words present in the corpus.

Return this file alongwith the function you wrote for generating the text.

"""



# Source Text is read from a text file

source_text = open('harry2.txt').read()

"""
 We will try to create a model which will learn to predict the next word in
 the sequence.
 For example,
 Consider text: Hello World! I am learning ASR.
   X    ,    Y
 Hello  ->  World!
 World! ->  I
 I      ->  am
 ...... and so on

"""


# integer encode text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([source_text])
encoded = tokenizer.texts_to_sequences([source_text])[0]

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

# create word -> word sequences
sequences = list()
for i in range(1, len(encoded)):
	sequence = encoded[i-1:i+1]
	sequences.append(sequence)
print('Total Sequences: %d' % len(sequences))


# split into X and y elements
sequences = array(sequences)
X = sequences[:,0]
y = sequences[:,1]

# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=1))
model.add(LSTM(64))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())


# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=80, verbose=2)
model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'



# This section is only for demonstration, remove this section when you finally submit the file on moodle.
in_text = 'the'
encoded = tokenizer.texts_to_sequences([in_text])[0]
encoded = array(encoded)
for i in range(100):
	encoded = tokenizer.texts_to_sequences([in_text])[0]
	encoded = array(encoded)
	pred = model.predict_classes(encoded, verbose=0)
	for word, index in tokenizer.word_index.items():
		if index == pred:
			print(word, end=' ')
			in_text = word



