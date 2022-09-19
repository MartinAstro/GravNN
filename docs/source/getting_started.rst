Getting Started
===============

Welcome to GravNN, the package responsible for generating and analyzing Physics-Informed Gravity Models.

To begin, we need to import some training data. To do this, turn to the Trajectories directory which contains a collection of different distributions from which training data can be generated.

Let's start with the simplest: `RandomDist`. `RandomDist` samples randomly in latitude, longitude, and altitude around a provided celestial body. 


.. code-block:: python

    from GravNN.Trajectories import RandomDist
    from GravNN.CelestialBodies.Planets import Earth

    trajectory = RandomDist(Earth(), [Earth().radius, Earth().radius + 420000.0], 10000)


The object `trajectory` will store the positions of all samples generated. Now we need to compute the accelerations of those positions. To do this we need to import a gravity model. Given that Earth's gravity field (and other large celestial bodies) are typically represented in spherical harmonics, we'll import the `SphericalHarmonics` model. 

.. code-block:: python

    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

    max_deg = 10
    model_file = planet.sh_file
    grav_model = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)



Note how the `SphericalHarmonics` model requires information about where the Stokes coefficients are located (`Earth().sh_file` includes the coefficients to degree and order 1000), the maximum degree of the expansion, and (optionally) the trajectory for which the accelerations / potentials must be computed. 

Now that the model has been configured, we will want to load in the accelerations and potentials. If these measurements have already been computed from a former experiment, they will simply be read in from a saved pickle file stored locally on your device. If not, they will be computed on-the-fly and saved accordingly. 

.. code-block:: python

    grav_model.load()
    x = grav_model.positions
    a = grav_model.accelerations

Now we have the simplest amount of training data for our experiment. Let's move on to training a Physics-Informed Neural Network (PINN) gravity model.

Training the Network
=============================

The first step in training the network is to configure all of the necessary hyperparameters and configuration variables prior to initialization.

The simplest option is to use one of the predefined configuration dictionaries stored in `GravNN.Networks.Configs`.

These dictionaries propagate through most of the package, so they contain variables responsible for answering "which network do you want to train", "how many data do you want to sample and from what distribution", "what learning rate and batch size do you want to use", and many more. 

We'll keep it simple for now and use the `get_earth_default_config` dictionary which will assume that we want to sample from a distribution that is random in latitude, longitude, and radius and that we train with a traditional densely connected network.

.. code-block:: python

    from GravNN.Networks.utils import configure_tensorflow
    from GravNN.Networks.Configs import get_earth_default_config

    config = get_earth_default_config()
    tf, mixed_precision = configure_tensorflow(config)
  
    # Standardize Configuration
    config = populate_config_objects(config)
    print(config)

    # Get data, network, optimizer, and generate model
    train_data, val_data, transformers = get_preprocessed_data(config)
    compute_input_layer_normalization_constants(config)
    dataset, val_dataset = configure_dataset(train_data, val_data, config)
    optimizer = configure_optimizer(config, mixed_precision)
    model = CustomModel(config)
    model.compile(optimizer=optimizer, loss="mse")

    # Train network
    callback = SimpleCallback()
    schedule = get_schedule(config)

    history = model.fit(
        dataset,
        epochs=config["epochs"][0],
        verbose=0,
        validation_data=val_dataset,
        callbacks=[callback, schedule],
    )
