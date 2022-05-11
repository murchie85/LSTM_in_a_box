from keras.utils import to_categorical
# Define Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding



def create_model(vocabulary_size, seq_len, Nunits=50,activationFunc='relu', ):
    
    model = Sequential()

    model.add(Embedding(input_dim=vocabulary_size, 
                        output_dim=seq_len, 
                        input_length=seq_len))

    model.add(LSTM(units=Nunits, return_sequences=True))

    model.add(LSTM(units=Nunits))

    model.add(Dense(units=Nunits, activation=activationFunc))

    model.add(Dense(units=vocabulary_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

    model.summary()
    return model


