import numpy as np
import sys
import hashlib

def labels2onehot(labels, classes):
    n_classes = len(classes)
    eye = np.eye(n_classes)
    y = []
    for sample in labels:
        curr = np.zeros(n_classes)
        for intent in sample:
            if intent not in classes:
                print('Warning: unknown intent {} detected'.format(intent))
                curr += eye[np.where(classes == 'unknown')[0]].reshape(-1)
            else:
                curr += eye[np.where(classes == intent)[0]].reshape(-1)
        y.append(curr)
    y = np.asarray(y)
    return y


def proba2labels(proba, confident_threshold, classes):
    y = []
    for sample in proba:
        to_add = np.where(sample > confident_threshold)[0]
        if len(to_add) > 0:
            y.append(classes[to_add])
        else:
            y.append([classes[np.argmax(sample)]])
    y = np.asarray(y)
    return y


def proba2onehot(proba, confident_threshold, classes):
    return labels2onehot(proba2labels(proba, confident_threshold, classes), classes)


def log_metrics(names, values, updates=None, mode='train'):
    sys.stdout.write("\r")  # back to previous line
    print("{} -->\t".format(mode), end="")
    if updates is not None:
        print("updates: {}\t".format(updates), end="")

    for id in range(len(names)):
        print("{}: {}\t".format(names[id], values[id]), end="")
    print(" ")  # , end='\r')
    return

def md5_hashsum(file_names):
    hash_md5 = hashlib.md5()
    for file_name in file_names:
        with open(file_name, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()
