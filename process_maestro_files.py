"""
The script in this file iterates through all of the .wav & .midi file pairs of the maestro dataset, computing transforms on the .wav files
& performing procesing to the .midi annotation files.  The resulting numpy arrays are uploaded to s3

"""
# !/usr/bin/env python3

from utils.maestro_data_utilities import generate_midi_wav_file_pairs_from_folder, process_wav_midi_pair, generate_annotation_matrices
import boto3
import os
import numpy as np

path = 'data/maestro-v3.0.0/2013'
processed_data_directory = 'data/meastro'

if not os.path.exists('data/maestro'):
    os.mkdir('data/maestro')

files = generate_midi_wav_file_pairs_from_folder(path)
first_wav = files[0][1]
first_midi = files[0][0]

# cqt, binary_annotation_matrix = process_wav_midi_pair(first_wav, first_midi)

s3 = boto3.client('s3')
BUCKET = 'dakobed-mir-data'
year = '2013'


for i, file_pair in enumerate(files):
    wav = file_pair[1]
    midi = file_pair[0]
    cqt, binary_annotation_matrix = process_wav_midi_pair(wav, midi)

    data_folder = 'data/maestro/{}/fileid{}'.format(year, i)

    if not os.path.exists('{}'.format(data_folder)):
        os.makedirs(data_folder)

    for file, array in [('{}/cqt.npy'.format(data_folder), cqt), ('{}/binary_annotation.npy'.format(data_folder), binary_annotation_matrix)]:
        np.save(file, arr=array)
        with open(file, "rb") as f:
            s3.upload_fileobj(f, BUCKET, file)
