"""
Command: ./__main__.py [command] [file1] [file2]
"""

import sys

from Harpon import Global_MNIST, le_xor_batch, le_xor_stochastic

def XOR():
    le_xor_stochastic(float(sys.argv[2]))

def MNIST():
    Global_MNIST(int(sys.argv[2]))

# Switcher
{
    'XOR': XOR,
    'MNIST': MNIST
}.get(sys.argv[1])()
