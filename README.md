<div align="center">
  <img src="docs/source/_static/logo.png">
</div>

# Welcome to the ML Gravity repo!

This repo contains the `GravNN` python package whose purpose is to train Physics-Informed Neural Networks to represent high-fidelity gravity fields. The package itself contains the tensorflow models, physics constraints, hyperparameter configurations, data generators, and visualization tools used in training such models. 

The `Examples` directory provides a set of minimal example scripts that leverage the core components of `GravNN` package to train a PINN and visualize some basic performance metrics. 

The `Scripts` directory provides a collection of python scripts and notebooks which make use of the components in `GravNN` for various research tasks. Given that, please note that this directory is under development and is currently used for exclusively for research -- not production. As such, not all scripts will work out of the box. Future releases will address these issues. If you do want to explore around in spite of this, I recommend starting from `Scripts/Networks/train_tensorflow_2.py` and traversing the GravNN package from there. 

Enjoy!