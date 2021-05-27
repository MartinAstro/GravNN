import tensorflow as tf


def _get_exp_decay_schedule(config):
    decay_rate = config.get("decay_rate")
    initial_learning_rate = config.get("learning_rate")
    decay_rate_epoch = config.get("decay_rate_epoch")
    decay_epoch_0 = config.get("decay_epoch_0")

    def exp_decay(epoch, lr):
        epoch0 = decay_epoch_0
        if epoch >= epoch0:
            return initial_learning_rate * (decay_rate) ** (
                (epoch - epoch0) / decay_rate_epoch
            )
        else:
            return lr

    return tf.keras.callbacks.LearningRateScheduler(exp_decay, verbose=0)


def _get_plateau_schedule(config):
    patience = config.get("patience")
    decay_rate = config.get("decay_rate")
    min_delta = config.get("min_delta")
    min_lr = config.get("min_lr")
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=patience,
        factor=decay_rate,
        min_delta=min_delta,
        verbose=1,
        min_lr=min_lr,
    )


def _get_cosine_decay_schedule(config):
    initial_learning_rate = config.get("learning_rate")
    decay_steps = config.get("decay_steps")
    alpha = config.get("alpha", 0.0)
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, alpha=alpha, name=None
    )


def _get_cosine_restart_decay_schedule(config):
    initial_learning_rate = config.get("learning_rate")
    first_decay_steps = config.get("first_decay_steps")
    return tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps,
        t_mul=config.get("t_mul", 2.0),
        m_mul=config.get("m_mul", 1.0),
        alpha=config.get("alpha", 0.0),
        name=None,
    )


def get_schedule(config):
    name = config["schedule_type"][0]

    if name == "exp_decay":
        schedule = _get_exp_decay_schedule(config)
    elif name == "plateau":
        schedule = _get_plateau_schedule(config)
    elif name == "cosine":
        schedule = _get_cosine_decay_schedule(config)
    elif name == "cosine_restart":
        schedule = _get_cosine_restart_decay_schedule(config)
    elif name == "none":
        schedule = tf.keras.callbacks.History()



    return schedule
