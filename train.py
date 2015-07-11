#!/usr/bin/env python

from __future__ import print_function
import argparse


parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('train', help='training text data')
arg('dict', help='dictionary file with word frequencies')
arg('output', help='file to save the model')
arg('--window', help='(max) window size', type=int, default=4)
arg('--min-freq', help='min. frequency of the word', type=int, default=20)
arg('--dim', help='dimensionality of representations', type=int, default=100)
arg('--prototypes', help='number of word prototypes', type=int, default=5)
arg('--alpha', help='prior probability of allocating a new prototype',
    type=float, default=0.1)

args = parser.parse_args()


from adagram import VectorModel, Dictionary


print('Building dictionary... ')
dictionary = Dictionary.read(args.dict, min_freq=args.min_freq)
print('Done! {} words.'.format(len(dictionary)))

model = VectorModel(freqs=dictionary.freqs,
    dim=args.dim, prototypes=args.prototypes, alpha=args.alpha)
