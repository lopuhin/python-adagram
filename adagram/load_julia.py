#!/usr/bin/env python
import argparse
import json
import os.path

from adagram import Dictionary, VectorModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'in_directory', help='Directory with model files in JSON format')
    parser.add_argument('out_file', help='File to save model to')
    args = parser.parse_args()

    with open(os.path.join(args.in_directory, 'vm.json'), 'rt') as f:
        vm_data = json.load(f)
    with open(os.path.join(args.in_directory, 'id2word.json'), 'rt') as f:
        id2word_data = json.load(f)
    assert len(id2word_data) == len(vm_data['frequencies'])
    dictionary = Dictionary(list(zip(id2word_data, vm_data['frequencies'])),
                            preserve_indices=True)
    model = VectorModel(dictionary, dim=len(vm_data['Out'][0]),
                        prototypes=len(vm_data['In'][0]),
                        alpha=vm_data['alpha'])
    model.counts[:] = vm_data['counts']
    model.Out[:] = vm_data['Out']
    model.In[:] = vm_data['In']
    model.path[:] = vm_data['path']
    model.path -= 1
    model.code[:] = vm_data['code']
    model.save(args.out_file)


if __name__ == '__main__':
    main()
