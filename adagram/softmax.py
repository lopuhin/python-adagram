from __future__ import print_function, division
import heapq


class HierarchicalSoftmaxNode(object):
    def __init__(self, parent=-1, branch=False):
        self.parent = parent
        self.branch = branch

    def is_root(self):
        return self.parent == -1

    def __repr__(self):
        return '<HierarchicalSoftmaxNode {} {}>'.format(
            self.parent, self.branch)


class HierarchicalOutput(object):
    def __init__(self, code, path):
        self.code = code
        self.path = path

    def __repr__(self):
        return '<HierarchicalOutput {} {}>'.format(self.code, self.path)


def softmax_path(nodes, N, idx):
    while True:
        node = nodes[idx]
        if node.is_root():
            break
        assert node.parent >= N
        yield node.parent - N, node.branch
        idx = node.parent


def build_huffman_tree(freqs):
    nodes = [HierarchicalSoftmaxNode() for _ in freqs]
    heap = zip(freqs, nodes)
    heapq.heapify(heap)

    def pop_initialize(parent, branch):
        freq, node = heapq.heappop(heap)
        node.parent = parent
        node.branch = branch
        return freq

    idx = len(nodes) - 1
    while len(heap) > 1:
        idx += 1
        node = HierarchicalSoftmaxNode()
        nodes.append(node)
        freq = pop_initialize(idx, True) + pop_initialize(idx, False)
        heapq.heappush(heap, (freq, node))
    assert len(heap) == 1
    return nodes


def convert_huffman_tree(nodes, N):
    outputs = []
    for idx in xrange(N):
        code = []
        path = []
        for n, branch in softmax_path(nodes, N, idx):
            code.append(branch)
            path.append(n)
        outputs.append(HierarchicalOutput(code, path))
    return outputs
