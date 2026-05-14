import glob
import pickle
from music21 import converter, instrument, note, chord

# dataset path
path = "dataset/*.midi"

notes = []

print("🎵 Starting preprocessing...")

for file in glob.glob(path):
    print("Parsing:", file)

    midi = converter.parse(file)

    parts = instrument.partitionByInstrument(midi)

    if parts:  
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

# save notes
with open("notes.pkl", "wb") as f:
    pickle.dump(notes, f)

print("\n✅ Total Notes:", len(notes))
print("✅ Preprocessing Completed! notes.pkl created")