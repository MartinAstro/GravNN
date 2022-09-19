
from json import load
from ..Examples import train_network, load_network, analyze_network, plot_gravity_map

def test_run_examples():
    train_network()
    load_network()
    analyze_network()
    plot_gravity_map()

