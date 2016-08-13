Python-AdaGram
==============

Python-AdaGram is an implementation of AdaGram (adaptive skip-gram) for Python.
It borrows a lot of C code from the original AdaGram implementation in Julia
(https://github.com/sbos/AdaGram.jl). AdaGram was introduced in a paper by
Sergey Bartunov, Dmitry Kondrashkin, Anton Osokin and Dmitry Vetrov
at http://arxiv.org/abs/1502.07257.

**Note**: this is a work in progress: it used to work,
but it lacks multi-threading, tests and disambiguation.
If you have a more mature implementation or want to help,
please get in touch.

Install
-------

::

    $ pip install python-adagram


Usage
-----

Train a model from command line::

    $ adagram-train tokenized.txt out.pkl

Input corpus must be already tokenized, with tokens (usually words)
separated by spaces.
There are many options available, see ``adagram-train --help``.

Load model::

    >>> import adagram
    >>> vm = adagram.VectorModel.load('out.pkl')

Get sense probabilities for some word::

    >>> vm.sense_probs('apple')
    [0.341832, 0.658164]

Get sense neighbours::

    >>> vm.sense_neighbors('apple', 0)
    [('almond', 0, 0.70396507),
     ('cherry', 1, 0.69193166),
     ('plum', 0, 0.690269),
     ('apricot', 0, 0.6882005),
     ('orange', 3, 0.6739181),
     ('pecan', 0, 0.6662803),
     ('pomegranate', 0, 0.6580653)
     ('blueberry', 0, 0.6509351),
     ('pear', 0, 0.6484747),
     ('peach', 0, 0.6313036)]

    >>> vm.sense_neighbors('apple',  1)
    [('macintosh', 0, 0.79053026),
     ('iifx', 0, 0.71349466),
     ('iigs', 0, 0.7030192),
     ('computers', 0, 0.6952761),
     ('kaypro', 0, 0.6938647),
     ('ipad', 0, 0.6914306),
     ('pc', 3, 0.6801078),
     ('ibm', 0, 0.66797054),
     ('powerpc-based', 0, 0.66319686),
     ('ibm-compatible', 0, 0.66120595)]

