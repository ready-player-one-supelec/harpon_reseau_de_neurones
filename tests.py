import numpy as np
from network_base import save_network, load_network, random_w_b

def test_network_dump():
    network = [4, 2, 1]
    weights, bias = random_w_b([1, 1, 1, 1], network)
    save_network(network, weights, bias, "test_dump.data")
    l_network, l_weights, l_bias = load_network("test_dump.data")
    assert np.array_equal(l_network, network)
    for i in range(len(weights)):
        assert np.allclose(l_weights[i], weights[i])
    for i in range(len(bias)):
        assert np.allclose(l_bias[i], bias[i])

if __name__ == '__main__':
    test_network_dump()
