import jams
import numpy as np
import os
import librosa


def jam_to_notes_matrix(jam_file):
    """
    This function processes the annotation found in a .jam file into a matrix of notes, with a dimension for each string
    of the guitar.
    :param jam_file: .jam file
    :return: matrix of notes
    """
    annotation = (jams.load(jam_file)).annotations
    # annotation['value'] = annotation['value'].apply(round_midi_numers)
    lowE = annotation[1]
    A = annotation[3]
    D = annotation[5]
    G = annotation[7]
    B = annotation[9]
    highE = annotation[11]
    notes = []
    for i, string in enumerate([lowE, A, D, G, B, highE]):
        for datapoint in string.data:
            notes.append([datapoint[0], datapoint[1], datapoint[2], i])
    notes.sort(key=lambda x: x[0])
    return np.asarray(notes, dtype=np.float32)


def notes_matrix_to_annotation(notes, nframes):
    """
    This function performs the inverse of the jam_to_notes_matrix, generating an annotation from a notes matrix.
    :param notes:
    :param nframes:
    :return:
    """
    binary_annotation_matrix = np.zeros((48, nframes))
    full_annotation_matrix = np.zeros((48, nframes, 6))
    for note in notes:
        starting_frame = librosa.time_to_frames(note[0])
        duration_frames = librosa.time_to_frames(note[1])
        ending_frame = starting_frame + duration_frames
        note_value, string = int(note[2]) - 35, int(note[3])
        binary_annotation_matrix[note_value, starting_frame:ending_frame] = 1
        full_annotation_matrix[note_value, starting_frame:ending_frame, string] = 1
    return binary_annotation_matrix, full_annotation_matrix


def annotation_audio_file_paths(audio_dir='data/guitarset/audio/', annotation_dir='data/guitarset/annotations'):
    """
    This function creates a list of tuples, associating each .wav file with it's .jam file.

    :param audio_dir: directory in which the .wav files are saved
    :param annotation_dir: directory in which the .jam annotation files are saved
    :return:
    """

    audio_files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, file) for file in audio_files]
    annotation_files = os.listdir(annotation_dir)
    annotation_files = [os.path.join(annotation_dir, file) for file in annotation_files]

    file_pairs = []
    for annotation in annotation_files:
        annotation_file = annotation.split('/')[-1].split('.')[0]
        for audio_file in audio_files:
            if audio_file.split('/')[-1].split('.')[0][:-4] == annotation_file:
                file_pairs.append((audio_file, annotation))
    return file_pairs
