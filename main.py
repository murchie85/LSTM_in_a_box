# enable conda to get this working 
import spacy
from processData import *
from _myModels import *
from pickle import dump
from keras.models import load_model

len_0 = 25
chosenEpoch = 20 # 0-1000
chosenBatch = 128 # 0 to 256
load=False
print('Sentence split Len: {} Epochs: {} Batches: {} '.format(len_0,chosenEpoch,chosenBatch))

#--------------------------------------------------------
#                LETTER ARRAY FROM FILES                #
#--------------------------------------------------------

tokenizerPath = 'savedModels/tokenizer'
modelPath     = 'savedModels/model.h5'
mypath ='/Users/adammcmurchie/code/2022/abstractLSTM/scripts/pinkrap'
fullData = letterArrayFromTextFiles(mypath,printme=True)

#--------------------------------------------------------
#                      TOKENIZE                         #
#--------------------------------------------------------


# Load language model. 
nlp = spacy.load('en_core_web_sm', disable = ['parser', 'tagger', 'ner'])
# Get tokens.
tokens = get_tokens(fullData,nlp)


#--------------------------------------------------------
#              BUILD SEQUENCE ARRAY                     #
#--------------------------------------------------------



print('Tokens/feature example is: \n{}\n'.format(str(tokens[0:len_0])))
print('Target or label is {}\n'.format(str(tokens[len_0:len_0 + 1])))
train_len = len_0 + 1
text_sequences = []
for i in range(train_len, len(tokens)):
    # Construct sequence.
    seq = tokens[i - train_len: i]
    # Append.
    text_sequences.append(seq)




#--------------------------------------------------------
#             WORD 2 VEC WITH KERAS                     #
#--------------------------------------------------------

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)


# Get numeric sequences.
sequences = tokenizer.texts_to_sequences(text_sequences)


print('******Please verify these match******')
print(tokenizer.index_word[int(sequences[0][0])])
print(fullData.split(' ')[0])
vocabulary_size = len(tokenizer.word_counts)
print('vocab size: ' + str(vocabulary_size))


# Pass sequences to np array so we can use it in our xy split
sequences = np.array(sequences)



#--------------------------------------------------------
#                   MODEL DEFINITION                    #
#--------------------------------------------------------


# select all but last word indices.
X = sequences[:, :-1]
seq_len = X.shape[1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=(vocabulary_size + 1))

print('printing X array: \n{}'.format(X))
print('printing y array: \n{}'.format(y))

model = create_model(vocabulary_size=(vocabulary_size + 1), seq_len=seq_len)


#--------------------------------------------------------
#                   MODEL FITTING                       #
#--------------------------------------------------------

if(load==False):
	print('fitting model please wait....')
	model.fit(x=X, y=y, batch_size=chosenBatch, epochs=chosenEpoch, verbose=1)


	# Produce loss and accuracy values
	loss, accuracy =  model.evaluate(x=X, y=y)
	print(f'Loss: {loss}\nAccuracy: {accuracy}')


	# Save Model 
	dump(tokenizer, open(tokenizerPath, 'wb'))  
	model.save(modelPath)
else:
	model = load_model('/Users/adammcmurchie/code/2022/abstractLSTM/savedModels/model.h5') 
	#with open(tokenizerPath, 'rb') as handle: tokenizer = pickle.load(handle)

#--------------------------------------------------------
#                  PREDICTIONS                          #
#--------------------------------------------------------

print('GENERATING FITS')



print("Random Seed")
sample_text = ' '.join(tokens[100:150])
seed_text   = sample_text[0:100]
generated_text = generate_text(model=model, tokenizer=tokenizer,seq_len=seq_len, seed_text=seed_text, num_gen_words=40)
print('Seed: {}  Generated Text: {} \n'.format(seed_text,generated_text))
print(seed_text + ' ' + generated_text + '...')


# SECOND MANUAL EXAMPLE  

seed_text = 'I eat a lot of ass, the whole ass I love the whole ass'
generated_text = generate_text(model=model, tokenizer=tokenizer,seq_len=seq_len, seed_text=seed_text, num_gen_words=40)
print('Seed: {}  Generated Text: {} \n'.format(seed_text,generated_text))
print(seed_text + ' ' + generated_text + '...')  


#--------------------------------------------------------
#                PREDICTIONS WITH TEMPERATURES          #
#--------------------------------------------------------


seed_text = 'I eat a lot of ass, the whole ass I love the whole ass'
temp = [0.9, 0.5, 0.1]
for tempValue in temp:
    print("Trying temperature at value : " + str(tempValue))
    generated_text = generate_textWithTemperature(model=model, tokenizer=tokenizer,seq_len=seq_len, seed_text=seed_text, num_gen_words=80, temperature=tempValue)
    print('Seed: {}  Generated Text: {} \n'.format(seed_text,generated_text))
    print(str(seed_text) + ' ' + str(generated_text) + ' ...')
    print('')
    print('--------')

