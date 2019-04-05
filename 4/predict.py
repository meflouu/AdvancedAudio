from keras.models import load_model

model = load_model('my_model.h5')

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