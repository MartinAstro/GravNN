import tensorflow as tf


def _get_exp_decay_schedule(config):
    """Learning rate decay schedule.

    TODO: make keywords required.

    Args:
        config (dict): hyperparameter and configuration variable dictionary.
        Needs to contain:
        decay_rate (float): Fraction of original learning rate between 0 and 1. (e.g. decay_rate of 0.5 causes the
        initial learning rate to decay as (1/2)**(i))
        decay_rate_epoch (int): Number of epochs necessary to cause the exponent to go up by 1
        decay_epoch_0 (int): the epoch after which the decay begins.

    Returns:
        LearningRateSchedule: The TF learning rate scheduler
    """
    decay_rate = config.get("decay_rate")[0]
    initial_learning_rate = config.get("learning_rate")[0]
    decay_rate_epoch = config.get("decay_rate_epoch")[0]
    decay_epoch_0 = config.get("decay_epoch_0")[0]

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
    """Learning rate schedule that decays after a plateau in val_loss.

    TODO: make keywords required.

    Args:
        config (dict): hyperparameter and configuration variable dictionary.
        Needs to contain:
        patience (int): how many consecutive epochs of the satisfied condition to wait before decaying the learning rate.
        decay_rate (float): Fraction of original learning rate between 0 and 1. (e.g. decay_rate of 0.5 causes the
        initial learning rate to decay as (1/2))
        min_delta (float): the difference in val_loss must exceed this amount, otherwise the epoch counts towards the patience quantity.
        min_lr (float): the minimum learning rate that can be decayed to.

    Returns:
        ReduceLROnPlateau: The TF learning rate scheduler
    """
    patience = config.get("patience")[0]
    decay_rate = config.get("decay_rate")[0]
    min_delta = config.get("min_delta")[0]
    min_lr = config.get("min_lr")[0]
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=patience,
        factor=decay_rate,
        min_delta=min_delta,
        verbose=1,
        min_lr=min_lr,
    )


def _get_cosine_decay_schedule(config):
    initial_learning_rate = config.get("learning_rate")[0]
    decay_steps = config.get("decay_steps")[0]
    alpha = config.get("alpha", 0.0)
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate,
        decay_steps,
        alpha=alpha,
        name=None,
    )


def _get_cosine_restart_decay_schedule(config):
    initial_learning_rate = config.get("learning_rate")[0]
    first_decay_steps = config.get("first_decay_steps")[0]
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
