from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Data import standardize_output
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros,Toutatis
from GravNN.Support.FrozenOrbitUtils import oe2cart_tf, sph2cart_tf

tf = configure_tensorflow()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hvplot.pandas
import pandas as pd
import hvplot.xarray
import xarray as xr
import panel as pn
from tqdm.notebook import tqdm
pn.extension()



def get_A_matrix(OE,model,mu):
    mu = tf.constant(mu, dtype=tf.float32, name='mu')
    a, e, i, w, O, M = tf.transpose(OE[:,].astype(np.float32))
    OE = tf.Variable(OE, dtype=tf.float32, name='orbit_elements')
    f = tf.zeros_like(a, dtype=tf.float32)
    f = tf.constant(0.0, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as outer_tape:
        outer_tape.watch(OE)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE) # Needs to be watched because it isn't a 'trainable variable' https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient
            r, v = oe2cart_tf(f, mu, OE)
            x = sph2cart_tf(r)
            u_pred = model.generate_potential(x)
        dUdOE = tape.gradient(u_pred, OE)
        
        b = a*tf.sqrt(1.-e**2)
        n = tf.sqrt(mu/a**3)
        
        dadt = 2.0/(n*a) * dUdOE[:,5]
        dedt = -b/(n*a**3*e)*dUdOE[:,3] + b**2/(n*a**4*e)*dUdOE[:,5]
        didt = -1.0/(n*a*b*tf.math.sin(i))*dUdOE[:,4] + tf.math.cos(i)/(n*a*b*tf.math.sin(i))*dUdOE[:,3]
        domegadt = -tf.math.cos(i)/(n*a*b*tf.math.sin(i))*dUdOE[:,2] + b/(n*a**3*e)*dUdOE[:,1]
        dOmegadt = 1.0/(n*a*b*tf.math.sin(i))*dUdOE[:,2]
        dMdt = -2.0/(n*a)*dUdOE[:,0] - b**2/(n*a**4*e)*dUdOE[:,1]
        dOEdt = tf.stack([dadt, dedt, didt, domegadt, dOmegadt, dMdt], axis=0)

    A = outer_tape.jacobian(dOEdt, OE)
    return tf.squeeze(A).numpy(), dOEdt.numpy()


def solve_nonlinear_system(OE, model, mu, iterations=10):
    b = np.zeros((6,1))
    x_n = OE
    J, dOEdt = get_A_matrix(OE, model, mu)
    # A*x = b
    
    for i in range(0, iterations):
        print(x_n)
        print(dOEdt)
        # u,s,v=np.linalg.svd(J)
        # Jinv=np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
        # x_n1 = x_n - np.dot(np.linalg.inv(p.dot(Jt,J)),np.dot(Jt,x_n))
        Jinv = np.linalg.pinv(J)
        x_n1 = (x_n.T - np.dot(Jinv, dOEdt)).T
        diff = np.linalg.norm(x_n1 - x_n)
        error = np.linalg.norm(np.abs(dOEdt.T))
        print("diff: %.4f" % diff)
        print("error: %.4f" % error)
        print(x_n1 - x_n)
        x_n = x_n1
        if diff < 1E-6:
            break
        J, dOEdt = get_A_matrix(x_n, model, mu)
    
    return x_n


def main():
    df_file = "Data/Dataframes/useless_072321_v1.data"
    df = pd.read_pickle(df_file)

    model_id = df["id"].values[0]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    mu = Toutatis().mu / 1000**3 # km^3/s^2
    # OE = np.array([[1000.0, 0.95, 2.0, 0.0, np.pi, 0.465]])
    OE = np.array([[1000.0, 0.05, 2.0, 0.0, np.pi, 0.465]])
    solve_nonlinear_system(OE, model, mu)

main()