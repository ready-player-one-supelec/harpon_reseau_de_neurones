import numpy as np
from network_base import save_network, load_network, random_w_b

def test_network_dump():
    network = [4, 2, 1]
    weights, bias = random_w_b([1, 1, 1, 1], network)
    save_network(network, weights, bias, "test_dump.data")
    l_network, l_weights, l_bias = load_network("test_dump.data")
    assert np.array_equal(l_network, network)
    for l_w, w in zip(l_weights, weights):
        assert np.allclose(l_w, w)
    for l_b, b in zip(l_bias, bias):
        assert np.allclose(l_b, b)

if __name__ == '__main__':
    test_network_dump()
