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


class LPE():
    def __init__(self, model, config, mu):
        self.model = model
        self.config = config 
        self.mu = tf.constant(mu, dtype=tf.float32, name='mu')

    def __call__(self, OE):
        a, e, i, w, O, M = tf.transpose(OE[:,].astype(np.float32))
        OE = tf.Variable(OE, dtype=tf.float32, name='orbit_elements')
        f = tf.zeros_like(a, dtype=tf.float32)
        f = tf.constant(0.0, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(OE) # Needs to be watched because it isn't a 'trainable variable' https://stackoverflow.com/questions/56916313/tensorflow-2-0-doesnt-compute-the-gradient
            r, v = oe2cart_tf(f, self.mu, OE)
            x = sph2cart_tf(r)
            u_pred = self.model.generate_potential(x)
        dUdOE = tape.gradient(u_pred, OE)

        # mu = tf.ones_like(a)*self.mu

        b = a*np.sqrt(1.-e**2)
        n = np.sqrt(self.mu/a**3)
        
        dOEdt = {
            'dadt' : 2.0/(n*a) * dUdOE[:,5],
            'dedt' : -b/(n*a**3*e)*dUdOE[:,3] + b**2/(n*a**4*e)*dUdOE[:,5],
            'didt' : -1.0/(n*a*b*np.sin(i))*dUdOE[:,4] + np.cos(i)/(n*a*b*np.sin(i))*dUdOE[:,3],
            'domegadt' : -np.cos(i)/(n*a*b*np.sin(i))*dUdOE[:,2] + b/(n*a**3*e)*dUdOE[:,1],
            'dOmegadt' : 1.0/(n*a*b*np.sin(i))*dUdOE[:,2],
            'dMdt' : -2.0/(n*a)*dUdOE[:,0] - b**2/(n*a**4*e)*dUdOE[:,1]
        }
       
        return dOEdt
    


def main():
    df_file = "Data/Dataframes/useless_072321_v1.data"
    df = pd.read_pickle(df_file)

    model_id = df["id"].values[0]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    mu = Toutatis().mu / 1000**3 # km^3/s^2
    solver = LPE(model, config, mu)

    OE_list = np.array([
                [100.0, 0.1, np.pi/3.0, 0.0, 0.0, 0.0],
                [100.0, 0.1, np.pi/3.0, 0.0, 0.0, 0.0], 
                [200.0, 0.2, np.pi/4.0, 0.0, 0.0, 0.0]
                ])


    columns = ['a', 'e', 'i', 'omega', 'Omega', 'M']
    x = pn.widgets.Select(value='a', options=columns, name='OE1')
    y = pn.widgets.Select(value='e', options=columns, name='OE2')

    planet = Toutatis()
    r_min = planet.radius
    r_max = planet.radius + 5000.

    def OE_to_dOEdt(key):
        return 'd' + key + 'dt'

    def dashboard(OE1, OE2):
        OE1 = x.value
        OE2 = y.value

        range_slider_dict = {
            'a' : pn.widgets.EditableRangeSlider(name='semi-major axis', start=r_min, end=r_max, step=10, value=(r_min, r_max)),
            'e' : pn.widgets.EditableRangeSlider(name='eccentricity', start=0.0, end=1.0, step=0.05, value=(0.0, 1.0)),
            'i' : pn.widgets.EditableRangeSlider(name='inclination', start=0, end=np.pi, step=0.05, value=(0, np.pi/4)),
            'omega' : pn.widgets.EditableRangeSlider(name='arg of periapsis', start=0.0, end=2*np.pi, step=0.05, value=(0, 2*np.pi)),
            'Omega' : pn.widgets.EditableRangeSlider(name='longitude of ascending node', start=0.0, end=2*np.pi, step=0.05, value=(0, 2*np.pi)),
            'M' : pn.widgets.EditableRangeSlider(name='mean anamoly', start=0.0, end=2*np.pi, step=0.05, value=(0, 2*np.pi))
        }
    
        float_slider_dict = {
            'a' : pn.widgets.EditableFloatSlider(name='semi-major axis', start=r_min, end=r_max, step=10, value=r_max),
            'e' : pn.widgets.EditableFloatSlider(name='eccentricity', start=0.0, end=1.0, step=0.05, value=0.3),
            'i' : pn.widgets.EditableFloatSlider(name='inclination', start=0, end=np.pi, step=0.05, value=np.pi/4),
            'omega' : pn.widgets.EditableFloatSlider(name='arg of periapsis', start=0.0, end=2*np.pi, step=0.05, value=0.0),
            'Omega' : pn.widgets.EditableFloatSlider(name='longitude of ascending node', start=0.0, end=2*np.pi, step=0.05, value=0.0),
            'M' : pn.widgets.EditableFloatSlider(name='mean anamoly', start=0.0, end=2*np.pi, step=0.05, value=0.0)
        }
        
        sliders = []
        for key, value in range_slider_dict.items():
            if key == OE1 or key == OE2:
                sliders.append(range_slider_dict[key])
            else:
                sliders.append(float_slider_dict[key])
        
        sliders.append(pn.widgets.EditableRangeSlider(name='color', start=-100.0, end=100.0, step=0.01, value=(-10.0, 10.0)))

        int_input = pn.widgets.IntInput(name='density', value=100, step=1, start=0, end=1000)
        OE_select = pn.widgets.Select(name='dOEdt', options=['a', 'e', 'i', 'omega', 'Omega', 'M'], value='i')

        def image_plot(OE_select, a_slider, e_slider, i_slider, o_slider, O_slider, M_slider, c_slider, int_input):
            OE = {
                'a': a_slider,
                'e': e_slider,
                'i': i_slider,
                'omega': o_slider,
                'Omega': O_slider,
                'tau': M_slider
            }
            keys = []
            for key, value in OE.items():
                if type(value) == type(()):
                    keys.append(key)
            N = int_input
            X = np.linspace(OE[keys[0]][0], OE[keys[0]][1], N)
            Y = np.linspace(OE[keys[1]][0], OE[keys[1]][1], N)
            XX, YY = np.meshgrid(X, Y)

            dUdOE = []
            dOE = []
            
            OE_list = []
            for idx in tqdm(range(len(XX)*len(YY))):
                i = idx // len(XX)
                j = idx % len(XX)

                OE[keys[0]] = XX[i,j]
                OE[keys[1]] = YY[i,j]
                OE_inst = [OE['a'], OE['e'], OE['i'], OE['omega'], OE['Omega'], OE['tau']] 
                OE_list.append(OE_inst)
            
            OE_list = np.array(OE_list)
                
            dOEdt = solver(OE_list)
            dOE = dOEdt['d' + OE_select + 'dt'].numpy()

            dOEdt_grid = np.zeros((len(X), len(Y)))
            for i in range(len(X)):
                for j in range(len(Y)):
                    idx = i*len(Y) + j
                    dOEdt_grid[i,j] = dOE[idx]

            ds = xr.DataArray(np.transpose(dOEdt_grid), dims=(keys[0], keys[1]), coords={keys[0] : X, keys[1] : Y})
            image = ds.hvplot.image(title='d' + OE_select + 'dt',clim=c_slider)
            return image
        
        image = pn.bind(image_plot, OE_select, *tuple(sliders), int_input)

        return pn.Row(pn.Column(x,y,OE_select,*tuple(sliders), int_input), image)


    pn.panel(pn.bind(dashboard, x, y)).servable()
# generate_sliders(x,y)

main()