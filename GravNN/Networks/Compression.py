import tensorflow as tf
import tensorflow_model_optimization as tfmot

def physics_constraints(model, x, PINN=False):
    # PINN Constraints
    if PINN:
        with tf.GradientTape() as tape:
            tape.watch(x)
            U_pred = model(x, training=True)
        a_pred = -tape.gradient(U_pred, x)
    else:  
        a_pred = model(x, training=True)
        U_pred = tf.constant(0.0)#None
    return U_pred, a_pred

#@tf.function
def pruning_training_loop(prunable_model, dataset, config):
    prunable_model.compile(loss='mse', optimizer='adam')

    batches = 1
    unused_arg = -1

    step_callback = tfmot.sparsity.keras.UpdatePruningStep()
    step_callback.set_model(prunable_model)
    step_callback.on_train_begin() # run pruning callback
    for _ in range(config['epochs'][0]):
        for x,y in dataset:
            step_callback.on_train_batch_begin(batch=unused_arg) # run pruning callback
            with tf.GradientTape() as tape:
                U_pred, a_pred = physics_constraints(prunable_model, x, config['PINN_flag'][0])
                loss_result = prunable_model.compiled_loss(y, a_pred)
            gradients = tape.gradient(loss_result, prunable_model.trainable_variables)
            prunable_model.optimizer.apply_gradients(zip(gradients, prunable_model.trainable_variables))
        step_callback.on_epoch_end(batch=unused_arg) # run pruning callback

    return prunable_model