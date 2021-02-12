import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.maestro_data_utilities import generate_midi_wav_file_pairs_from_folder

path = 'data/maestro-v3.0.0/2013'
processed_data_directory = 'data/meastro'

if not os.path.exists('data/maestro'):
    os.mkdir('data/maestro')

files = generate_midi_wav_file_pairs_from_folder(path)

first_wav_file = files[0][1]

y, sr = librosa.load(first_wav_file)
tensecondsamples = 22050 * 5

first_ten_seconds_y = y[:tensecondsamples]

# D = librosa.amplitude_to_db(librosa.stft(first_ten_seconds_y), ref=np.max)
# plt.figure(figsize=(15, 5))
# librosa.display.specshow(D,x_axis='time', y_axis='log', cmap='gray_r')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Log-frequency power spectrogram')
#
#
cqt = librosa.amplitude_to_db(librosa.cqt(first_ten_seconds_y, sr=sr), ref=np.max)
plt.figure(figsize=(15, 5))
librosa.display.specshow(cqt, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (Hz)')
plt.show()

cqt = librosa.cqt(y, sr=sr)
magnitude, phase = librosa.magphase(cqt)
librosa.display.specshow(magnitude, x_axis = 'time', y_axis='cqt_note')
plt.title('Magnitude of complex-valued spectogram')
plt.show()

cqt_raw = librosa.core.cqt(first_ten_seconds_y, sr=sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('A0'), norm=1)
magnitude, phase = librosa.magphase(cqt_raw)
librosa.display.specshow(magnitude, x_axis = 'time', y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Magnitude of complex-valued spectogram')
plt.show()
