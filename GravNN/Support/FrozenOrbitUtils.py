import numpy as np

from GravNN.Networks.utils import configure_tensorflow

tf = configure_tensorflow()


def trad2cart_tf(f, mu, OE):
    a = OE[:, 0]
    e = OE[:, 1]
    i = OE[:, 2]
    omega = OE[:, 3]
    Omega = OE[:, 4]
    OE[:, 5]

    p = a * (1 - e**2)

    # e_tensor, e_mag = tf.linalg.normalize(e,axis=1)
    e_mag = e

    cf = tf.math.cos(f)
    sf = tf.math.sin(f)

    r_eHatTerm = p * cf / (1.0 + e_mag * cf)
    r_ePerpHatTerm = p * sf / (1.0 + e_mag * cf)

    v_eHatTerm = tf.math.sqrt(mu / p) * -sf
    v_ePerpHatTerm = tf.math.sqrt(mu / p) * (e_mag + cf)

    cw = tf.math.cos(omega)
    sw = tf.math.sin(omega)

    cO = tf.math.cos(Omega)
    sO = tf.math.sin(Omega)

    ci = tf.math.cos(i)
    si = tf.math.sin(i)

    r_nHat = r_eHatTerm * tf.stack([cw, sw], 0) + r_ePerpHatTerm * tf.stack(
        [-1.0 * sw, cw],
        0,
    )
    r_xyz = r_nHat[0, :] * tf.stack([cO, sO, tf.zeros_like(cO)], 0) + r_nHat[
        1,
        :,
    ] * tf.stack([-1.0 * ci * sO, ci * cO, si], 0)

    v_nHat = v_eHatTerm * tf.stack([cw, sw], 0) + v_ePerpHatTerm * tf.stack(
        [-1.0 * sw, cw],
        0,
    )
    v_xyz = v_nHat[0] * tf.stack([cO, sO, tf.zeros_like(cO)], 0) + v_nHat[1] * tf.stack(
        [-1.0 * ci * sO, ci * cO, si],
        0,
    )

    return tf.transpose(r_xyz), tf.transpose(v_xyz)


def sph2cart_tf(r_vec):
    r = r_vec[:, 0]  # [0, inf]
    theta = r_vec[:, 1] * np.pi / 180.0  # [0, 360]
    phi = r_vec[:, 2] * np.pi / 180.0  # [0, 180]

    x = r * tf.math.sin(phi) * tf.math.cos(theta)
    y = r * tf.math.sin(phi) * tf.math.sin(theta)
    z = r * tf.math.cos(phi)

    return tf.stack([x, y, z], 1)


def main():
    f = 0.0
    mu = 1.1e14
    OE = tf.convert_to_tensor([[1.1e7, 0.1, np.pi / 3, np.pi / 3, 0.0, 0.0]])

    cart_state = trad2cart_tf(f, mu, OE)
    print(cart_state[:, 0:3])
    print(cart_state[:, 3:6])


if __name__ == "__main__":
    main()
