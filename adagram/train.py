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
arg('--context-cut', help='randomly reduce size of the context',
    action='store_true')
arg('--epochs', help='number of epochs to train', type=int, default=1)


args = parser.parse_args()


from adagram.model import VectorModel, Dictionary
from adagram.gradient import inplace_train


print('Building dictionary... ')
dictionary = Dictionary.read(args.dict, min_freq=args.min_freq)
print('Done! {} words.'.format(len(dictionary)))

vm = VectorModel(frequencies=dictionary.frequencies,
    dim=args.dim, prototypes=args.prototypes, alpha=args.alpha)

inplace_train(vm, dictionary, args.train, args.window,
    context_cut=args.context_cut, epochs=args.epochs)
