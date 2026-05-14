import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.utils import to_categorical

# load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

print("Total notes:", len(notes))

# unique notes
pitchnames = sorted(set(notes))

# mapping
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 100

network_input = []
network_output = []

# create sequences
for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)

# reshape input
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
network_input = network_input / float(len(pitchnames))

network_output = to_categorical(network_output)

# MODEL BUILDING
model = Sequential()

model.add(LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(256))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(Dropout(0.3))

model.add(Dense(len(pitchnames)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print("✅ Model Created Successfully!")

# save model architecture
model.save("music_model.h5")

print("✅ Model Saved as music_model.h5")