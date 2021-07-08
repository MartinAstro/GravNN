from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Data import standardize_output
tf = configure_tensorflow()
import numpy as np
import pandas as pd

def oe2cart_tf(f, mu, OE):

    a = OE[0]
    e = OE[1]
    i = OE[2]
    omega = OE[3]
    Omega = OE[4]
    tau = OE[5]

    p = a*(1-e**2)

    e_tensor, e_mag = tf.linalg.normalize(e)

    cf = tf.math.cos(f)
    sf = tf.math.sin(f)

    r_eHatTerm = p*cf/(1.0+e_mag*cf)
    r_ePerpHatTerm = p*sf/(1.0+e_mag*cf)

    v_eHatTerm = tf.math.sqrt(mu/p)*-sf
    v_ePerpHatTerm = tf.math.sqrt(mu/p)*(e_mag+cf)

    cw = tf.math.cos(omega)
    sw = tf.math.sin(omega)

    cO = tf.math.cos(Omega)
    sO = tf.math.sin(Omega)

    ci = tf.math.cos(i)
    si = tf.math.sin(i)

    r_nHat = r_eHatTerm*tf.stack([cw,  sw], 0) + r_ePerpHatTerm*tf.stack([-1.0*sw, cw], 0)
    r_xyz = r_nHat[0]*tf.stack([cO, sO, 0.0], 0) + r_nHat[1]*tf.stack([-1.0*ci*sO, ci*cO, si], 0)

    v_nHat = v_eHatTerm*tf.stack([cw,  sw], 0) + v_ePerpHatTerm*tf.stack([-1.0*sw, cw], 0)
    v_xyz = v_nHat[0]*tf.stack([cO, sO, 0.0], 0) + v_nHat[1]*tf.stack([-1.0*ci*sO, ci*cO, si], 0)

    return r_xyz, v_xyz


class LPE():
    def __init__(self, model, config, mu):
        self.model = model
        self.config = config 
        x_transformer = config['x_transformer'][0]
        u_transformer = config['u_transformer'][0]

        x_scaler = x_transformer.data_range_
        u_scaler = u_transformer.data_range_
    
        self.x0 = tf.constant(x_scaler, dtype=tf.float32, name='x_scale')
        self.u0 = tf.constant(u_scaler, dtype=tf.float32, name='u_scale')

        self.mu = tf.constant(mu, dtype=tf.float32, name='mu')

    def __call__(self, OE):
        a, e, i, w, O, M = OE
        OE = tf.Variable(OE, dtype=tf.float32, name='orbit_elements')
        f = tf.constant(0.0, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE) # Needs to be watched because it isn't a 'trainable variable' https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient
            r, v = oe2cart_tf(f, self.mu, OE)
            r_scaled = tf.reshape(r/self.x0, shape=(1,3))
            y_hat = self.model(r_scaled)#.astype('float32'))
            u_pred, a_pred, laplace_pred, curl_pred = standardize_output(y_hat, self.config)
            u = u_pred*self.u0
        dUdOE = tape.gradient(u, OE)


        b = a*np.sqrt(1.-e**2)
        n = np.sqrt(self.mu/a**3)
        
        dadt = 2.0/(n*a) * dUdOE[5]
        dedt = -b/(n*a**3*e)*dUdOE[3] + b**2/(n*a**4*e)*dUdOE[5]
        didt = -1.0/(n*a*b*np.sin(i))*dUdOE[4] + np.cos(i)/(n*a*b*np.sin(i))*dUdOE[3]
        dwdt = -np.cos(i)/(n*a*b*np.sin(i))*dUdOE[2] + b/(n*a**3*e)*dUdOE[1]
        dOdt = 1.0/(n*a*b*np.sin(i))*dUdOE[2]
        dMdt = -2.0/(n*a)*dUdOE[0] - b**2/(n*a**4*e)*dUdOE[1]

        return [dadt, dedt, didt, dwdt, dOdt, dMdt]

def main():

    df_file = "Data/Dataframes/useless_061121.data"
    df = pd.read_pickle(df_file)
    idx = 0

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    a = 6378.0+420.0
    e = 0.1
    i = np.pi/4.0
    omega = 0.0
    Omega = 0.0
    tau = 0.0
    OE = [a, e, i, omega, Omega, tau]
    mu = 3.986004418*10**14 / 1000**3 # km^3/s^2

    solver = LPE(model, config, mu)

    N = 20
    omega = np.linspace(0.0, 2.0*np.pi, N)
    e = np.linspace(0.05, 0.99, N)
    XX, YY = np.meshgrid(omega, e)

    dUdX = []
    dUdY = []
    for i in range(len(XX)):
        dX = []
        dY = []
        for j in range(len(YY)):
            x = XX[i,j]
            y = YY[i,j]
            OE = [a, y, i, x, Omega, tau]
            output = solver(OE)
            dX.append(output[3].numpy())
            dY.append(output[1].numpy())
        dUdX.append(dX)
        dUdY.append(dY)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.contourf(XX,YY,dUdX, levels=np.linspace(-0.00005, 0.00015,30))
    #plt.plot(dUda)
    plt.colorbar()
    plt.show()
    print(output)



if __name__ == "__main__":
    main()