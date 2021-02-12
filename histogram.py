import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.maestro_data_utilities import generate_midi_wav_file_pairs_from_folder, extract_notes


path = 'data/maestro-v3.0.0/2013'
processed_data_directory = 'data/meastro'

if not os.path.exists('data/maestro'):
    os.mkdir('data/maestro')

files = generate_midi_wav_file_pairs_from_folder(path)

midi_file = files[0][0]

notes = extract_notes(midi_file)
note_durations = [float(note[3]) for note in notes]

first_two_hundred_note_durations = np.array(note_durations[:200])

histogram = np.histogram(first_two_hundred_note_durations)


plt.hist(first_two_hundred_note_durations, bins=50)
plt.title('Note Duration Histogram')
plt.xlabel = 'Note Duration (Seconds)'
plt.ylabel = 'Count'

plt.show()