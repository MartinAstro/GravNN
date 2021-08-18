from GravNN.Networks.utils import configure_tensorflow
from GravNN.Networks.Model import load_config_and_model
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros, Toutatis
tf = configure_tensorflow()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def sph2cart_tf(r_vec):
    r = r_vec[0] #[0, inf]
    theta = r_vec[1] * np.pi / 180.0 # [0, 360]
    phi = r_vec[2]* np.pi / 180.0 # [0, 180]

    x = r*tf.math.sin(phi)*tf.math.cos(theta)
    y = r*tf.math.sin(phi)*tf.math.sin(theta)
    z = r*tf.math.cos(phi)

    return tf.stack([[x,y,z]],0)
class LPE():
    def __init__(self, model, config, mu):
        self.model = model
        self.config = config 
        self.mu = tf.constant(mu, dtype=tf.float32, name='mu')

    def __call__(self, OE):
        a, e, i, w, O, M = OE
        OE = tf.Variable(OE, dtype=tf.float32, name='orbit_elements')
        f = tf.constant(0.0, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE) # Needs to be watched because it isn't a 'trainable variable' https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient
            r, v = oe2cart_tf(f, self.mu, OE)
            x = sph2cart_tf(r)
            u_pred = self.model.generate_potential(x)
        dUdOE = tape.gradient(u_pred, OE)

        b = a*np.sqrt(1.-e**2)
        n = np.sqrt(self.mu/a**3)
        
        dOEdt = {
            'dadt' : 2.0/(n*a) * dUdOE[5],
            'dedt' : -b/(n*a**3*e)*dUdOE[3] + b**2/(n*a**4*e)*dUdOE[5],
            'didt' : -1.0/(n*a*b*np.sin(i))*dUdOE[4] + np.cos(i)/(n*a*b*np.sin(i))*dUdOE[3],
            'domegadt' : -np.cos(i)/(n*a*b*np.sin(i))*dUdOE[2] + b/(n*a**3*e)*dUdOE[1],
            'dOmegadt' : 1.0/(n*a*b*np.sin(i))*dUdOE[2],
            'dMdt' : -2.0/(n*a)*dUdOE[0] - b**2/(n*a**4*e)*dUdOE[1]
        }
       
        return dOEdt


def generate_2d_dOE_grid(OE, N, solver):
    keys = []
    for key, value in OE.items():
        if type(value) == type([]):
            keys.append(key)
    
    X = np.linspace(OE[keys[0]][0], OE[keys[0]][1], N)
    Y = np.linspace(OE[keys[1]][0], OE[keys[1]][1], N)
    XX, YY = np.meshgrid(X, Y)

    dUdX = []
    dUdY = []
    for i in range(len(XX)):
        dX = []
        dY = []
        for j in range(len(YY)):
            OE[keys[0]] = XX[i,j]
            OE[keys[1]] = YY[i,j]
            OE_inst = [OE['a'], OE['e'], OE['i'], OE['omega'], OE['Omega'], OE['tau']]
            dOEdt = solver(OE_inst)
            dX.append(dOEdt['d' + keys[0] + 'dt'].numpy())
            dY.append(dOEdt['d' + keys[1] + 'dt'].numpy())
        dUdX.append(dX)
        dUdY.append(dY)


    plt.figure()
    plt.contourf(XX,YY,dUdX, levels=np.linspace(-0.00005, 0.00015, 30))
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.title('d' + keys[0] + 'dt')
    plt.colorbar()

    plt.figure()
    plt.contourf(XX,YY,dUdY, levels=np.linspace(-0.1, 0.1, 30))
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.title('d' + keys[1] + 'dt')
    plt.colorbar()

    # plt.figure()
    # plt.imshow(dUdX)#, levels=np.linspace(-0.00005, 0.00015, 30))
    # plt.xlabel(keys[0])
    # plt.ylabel(keys[1])     
    # plt.title('d' + keys[0] + 'dt')
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(dUdY)#, levels=np.linspace(-0.00005, 0.00015, 30))
    # plt.xlabel(keys[0])
    # plt.ylabel(keys[1])
    # plt.title('d' + keys[1] + 'dt')
    # plt.colorbar()

    plt.show()
    print(dOEdt)

def main():

    #df_file = "Data/Dataframes/useless_061121.data"
    #mu = Earth().mu / 1000**3 # km^3/s^2
    
    df_file = "Data/Dataframes/eros_trajectory_v2.data"
    df_file = "Data/Dataframes/useless_072321_v1.data"


    # mu = Eros().mu / 1000**3 # km^3/s^2
    mu = Toutatis().mu / 1000**3
    df = pd.read_pickle(df_file)
    idx = 0

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    a = 6378.0+420.0 # Earth in LEO
    a = 20.0 # Eros ~16 km diameter

    solver = LPE(model, config, mu)
    N = 20

    OE_dict = {
        'a' : [16.0, 18.0],
        'e' : [0.2, 0.99],
        'i' : np.pi/3.0,
        'omega' : 0.0,
        'Omega' : 0.0,
        'tau' : 0.0
    }

    generate_2d_dOE_grid(OE_dict, N, solver)



if __name__ == "__main__":
    main()