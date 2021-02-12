import os
import boto3
import numpy as np
from mido import MidiFile, merge_tracks
from collections import defaultdict
from numpy import format_float_positional
import librosa
from librosa import time_to_frames


def process_wav_midi_pair(wav, midi):
    """
    This function processes a pair of .wav & .midi files, returning a cqt transformation of the audio file contained in the .wav file and a binary annotation matrix of the associated annotation from the .midi file.
    :param wav:
    :param midi:
    :return:
    """
    y, sr = load_wav_file(wav)
    annotation = extract_notes(midi)
    cqt = perform_cqt_transform(y, sr)
    binary_annotation_matrix = generate_annotation_matrices(annotation, cqt.shape[0])
    return cqt, binary_annotation_matrix


def load_wav_file(wav):
    y, sr = librosa.load(wav)
    return y, sr


def generate_midi_wav_file_pairs_from_folder(path):

    file_pairs = []

    files = os.listdir("{}".format(path))
    files.sort()
    for i in range(0, len(files), 2):
        if files[i][:-4] == files[i + 1][:-3]:
            file_pairs.append(('{}/{}'.format(path, files[i]) , '{}/{}'.format(path, files[i+1])))
    return file_pairs


def perform_cqt_transform(y,sr):
    cqt_raw = librosa.core.cqt(y, sr=sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('A0'),norm=1)
    magnitude, phase = librosa.magphase(cqt_raw)
    magphase_cqt = librosa.magphase(cqt_raw)
    cqt = magphase_cqt[0].T
    return cqt


def generate_midi_wav_file_pairs():
    path='data/maestro-v3.0.0'
    folders =['2013','2011','2006','2017','2015','2008', '2009', '2014', '2018', '2004']
    count = 0
    file_pairs = []

    for folder in folders:
        files = os.listdir("{}/{}".format(path,folder))
        files.sort()
        for i in range(0, len(files), 2):
            if files[i][:-4] == files[i + 1][:-3]:
                file_pairs.append(('{}/{}/{}'.format(path, folder, files[i]), '{}/{}/{}'.format(path, folder, files[i+1])))
    return file_pairs


def seconds_per_tick(mid):
    '''
    Returns the second per tick for the midi files
    :param mid:
    :return:
    '''
    ticks_per_quarter = mid.ticks_per_beat
    microsecond_per_quarter = mid.tracks[0][0].tempo
    microsecond_per_tick = microsecond_per_quarter / ticks_per_quarter
    seconds_per_tick = microsecond_per_tick / 1000000
    return float(format_float_positional(seconds_per_tick, 5))


def extract_notes(midi_file):
    '''
    Return a list of notes extracted from the MIDI file
    :param midi_file:
    :return:
    '''
    mid = MidiFile(midi_file)
    spt = seconds_per_tick(mid)
    current_time = 0
    current_notes = set()
    notes = []
    for msg in merge_tracks(mid.tracks):
        current_time += msg.time
        if msg.type == 'note_on':
            if msg.velocity != 0:
                current_notes.add((msg.note, current_time, msg.velocity))
            else:
                note = remove_note(current_notes, msg.note)
                notes.append((msg.note, spt * note[1], spt * current_time,
                              format_float_positional(spt * (current_time - note[1]), 2), note[2]))
    return notes


def generate_annotation_matrices(annotation, frames):
    '''
    This function will return a one hot encoded matrix of notes being played
    The annotation matrix will start w/ note 25 at index 0 and go up to note 100
    The highest and lowest values that I saw in the annotations seemed to be arounnd 29-96 so give a little leeway
    :return:
    '''
    annotation_matrix = np.zeros((84, frames))
    for note in annotation:
        starting_frame = time_to_frames(note[1])
        duration_frames = time_to_frames(note[2] - note[1])
        note_value = note[0]
        annotation_matrix[note_value - 25][starting_frame:starting_frame + duration_frames] = 1
    return annotation_matrix.T


def remove_note(notes_set, note_value):
    '''
    Removes a tuple from a set based on the value of it's first element.
    '''
    for note in notes_set:
        if note[0] == note_value:
            notes_set.remove(note)
            return note
