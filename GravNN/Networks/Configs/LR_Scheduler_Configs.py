def ReduceLrOnPlateauConfig():
    scheduler_config = {
        'schedule_type' : ['plateau'],
        "patience" : [2500],
        "decay_rate" : [0.9],
        "min_delta" : [0.0001],
        "min_lr" : [0.0001],
    }
    return scheduler_config

def ExpDecayConfig():
    scheduler_config = {
        "schedule_type" : ['exp_decay'],
        "decay_rate_epoch": [2500],
        "decay_epoch_0": [500],
        "decay_rate": [0.5],
    }
    return scheduler_config

        