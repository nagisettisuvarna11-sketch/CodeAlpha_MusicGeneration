import pickle
import numpy as np

from music21 import instrument, note, stream, chord
from keras.models import load_model

# load notes
with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

# unique notes
pitchnames = sorted(set(notes))

# mappings
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

sequence_length = 100

network_input = []

# prepare sequences
for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[n] for n in seq_in])

n_patterns = len(network_input)

# load trained model
model = load_model("music_model_trained.keras")

print("🎵 Generating 5 Music Files...")

# generate 5 songs
for song_num in range(1, 6):

    start = np.random.randint(0, len(network_input)-1)

    pattern = network_input[start]
    prediction_output = []

    # generate notes
    for note_index in range(500):

        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(len(pitchnames))

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]

        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    # convert to MIDI
    offset = 0
    output_notes = []

    for pattern in prediction_output:

        # chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    # save unique file
    output_file = f"generated_music_{song_num}.mid"

    midi_stream.write('midi', fp=output_file)

    print(f"✅ {output_file} created!")

print("🎉 All 5 AI music files generated successfully!")