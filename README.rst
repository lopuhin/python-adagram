Python-AdaGram
==============

Python-AdaGram is an implementation of AdaGram (adaptive skip-gram) for Python.
It borrows a lot of C code from the original AdaGram implementation in Julia
(https://github.com/sbos/AdaGram.jl). AdaGram was introduced in a paper by
Sergey Bartunov, Dmitry Kondrashkin, Anton Osokin and Dmitry Vetrov
at http://arxiv.org/abs/1502.07257.

**Note**: this is a work in progress: it used to work,
but it lacks multithreading, tests and also maybe disambiguation.
If you have a more mature implementation or want to help,
please get in touch.

Install
-------

::

    pip install python-adagram


Usage
-----

Train a model from command line::

    adagram-train tokenized.txt out.pkl

Input corpus must be already tokenized, with tokens (usually words)
separated by spaces.

Load model::

    import adagram
    vm = adagram.VectorModel.load('out.pkl')

