import csv
import os.path

import numpy as np

def num_to_array(num: int):
    arr = np.zeros(10)
    arr[num] = 1.0
    return arr

labels = []
pixels = []
file_path = "data/raw/dataNumbers.csv"

with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)

    for row in reader:
        label = num_to_array(int(row[0]))
        pixel_values = np.array([int(pixel) for pixel in row[1:]], dtype=np.uint8)
        labels.append(label)
        pixels.append(pixel_values)

labels = np.array(labels)
pixels = np.array(pixels)

directory = "data/processed"
np.save(os.path.join(directory, "labelsNumbers.npy"), labels)
np.save(os.path.join(directory, "pixelsNumbers.npy"), pixels)

def names_numbers(name: str):
    switcher = {
        "Low": -1.0,
        "Medium": 0.0,
        "High": 1.0
    }
    return switcher.get(name)

labels = []
factors = []
file_path = "data/raw/dataCancer.csv"

with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)

    for row in reader:

        label = names_numbers(row[-1])
        factor_values = np.array([int(factor) for factor in row[2:-1]], dtype=np.uint8)

        labels.append(label)
        factors.append(factor_values)

labels = np.array(labels)
factors = np.array(factors)

normalized_factors = (factors - factors.min(axis=0)) / (factors.max(axis=0) - factors.min(axis=0))

directory = "data/processed"
np.save(os.path.join(directory, "labelsCancer.npy"), labels)
np.save(os.path.join(directory, "factorsCancer.npy"), factors)

