import numpy as np
import tensorflow as tf


def cart2sph(X_cart):
    X_cart_transpose = tf.transpose(X_cart)
    X = X_cart_transpose[0, :]
    Y = X_cart_transpose[1, :]
    Z = X_cart_transpose[2, :]

    rad2deg = tf.constant(180.0 / np.pi, dtype=X_cart.dtype)
    offset = tf.constant(180.0, dtype=X_cart.dtype)

    r = tf.sqrt(X**2 + Y**2 + Z**2)
    theta = tf.math.atan2(Y, X) * rad2deg + offset  # [0, 360]
    phi = tf.math.atan2(tf.sqrt(X**2 + Y**2), Z) * rad2deg  # [0, 180]

    spheres = tf.transpose(tf.stack([r, theta, phi], 0))
    return spheres


def compute_projection_matrix(X_sphere):
    deg2rad = tf.constant(np.pi / 180.0, dtype=X_sphere.dtype)
    offset = tf.constant(np.pi, dtype=X_sphere.dtype)

    X_sphere_T = tf.transpose(X_sphere)
    theta = X_sphere_T[1, :] * deg2rad - offset
    phi = X_sphere_T[2, :] * deg2rad

    zero = tf.zeros_like(phi, dtype=X_sphere.dtype)

    r_hat = tf.stack(
        [tf.sin(phi) * tf.cos(theta), tf.sin(phi) * tf.sin(theta), tf.cos(phi)],
        0,
    )

    theta_hat = tf.stack(
        [tf.cos(phi) * tf.cos(theta), tf.cos(phi) * tf.sin(theta), -tf.sin(phi)],
        0,
    )

    phi_hat = tf.stack([-tf.sin(theta), tf.cos(theta), zero], 0)

    # a_B = [BN] @ a_N

    BN = tf.stack(
        [tf.transpose(r_hat), tf.transpose(theta_hat), tf.transpose(phi_hat)],
        axis=1,
    )
    return BN

    project_acc = tnp.zeros(accelerations.shape)
    for i in range(0, len(positions)):
        theta = positions[i, 1] * tnp.pi / 180.0 - tnp.pi
        phi = positions[i, 2] * tnp.pi / 180.0
        r_hat = tnp.array(
            [
                tnp.sin(phi) * tnp.cos(theta),
                tnp.sin(phi) * tnp.sin(theta),
                tnp.cos(phi),
            ],
        )
        theta_hat = tnp.array(
            [
                tnp.cos(phi) * tnp.cos(theta),
                tnp.cos(phi) * tnp.sin(theta),
                -tnp.sin(phi),
            ],
        )
        phi_hat = tnp.array([-tnp.sin(theta), tnp.cos(theta), 0.0])
        project_acc[i, :] = tnp.array(
            [
                tnp.dot(accelerations[i], r_hat),
                tnp.dot(accelerations[i], theta_hat),
                tnp.dot(accelerations[i], phi_hat),
            ],
        )


def convert_losses_to_sph(x, y, y_hat):
    x_sph = cart2sph(x)
    BN = compute_projection_matrix(x_sph)

    # x_B = tf.matmul(BN, tf.reshape(x, (-1,3,1)))
    # this will give ~[1, 1E-8, 1E-8]

    y_reshape = tf.reshape(y, (-1, 3, 1))
    y_hat_reshape = tf.reshape(y_hat, (-1, 3, 1))

    y = tf.matmul(BN, y_reshape)
    y_hat = tf.matmul(BN, y_hat_reshape)

    return y, y_hat
