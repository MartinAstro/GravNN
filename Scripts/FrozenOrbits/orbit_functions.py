def computeTrueAnomaly(t, mu, OE):
    e = OE[1]
    tau = OE[5]
    e_mag = np.linalg.norm(e)

    n = np.sqrt(mu)*(OE[0]**(-3.0/2.0))
    M = n*(t - tau)
    E = M

    # Given M, find E
    tol = 1e-11
    while np.abs(fFunc(M, E, e_mag)) > tol:
        f0 = fFunc(M,E,e_mag)
        dfdE = -1.0*(1.0 - e_mag*np.cos(E))
        deltaE = -f0/dfdE
        E = E + deltaE

    f_New = 2*np.arctan2(((1+e_mag)/(1-e_mag))**(1/2)*np.tan(E/2),1)
    return f_New #returns true anomaly

def cart2oe(r0, v0, t0, mu):
    #Calculate Orbital Elements
    z_hat = np.array([0.0, 0.0, 1.0])
    y_hat = np.array([0.0, 1.0, 0.0])
    x_hat = np.array([1.0, 0.0, 0.0])

    h = np.cross(r0, v0)
    h_hat = normalize(h)
    p = (np.dot(h, h)) / mu
    i = np.arccos(np.dot(h_hat, z_hat)) # if h_hat and z_hat are parallel i is undefined
    if i == 0 or i == np.pi:
        raise ValueError("h_hat and z_hat are parallel (i.e. i=0 or i=pi) which prevent the system from being well defined!")

    n_Omega_hat = np.cross(z_hat, h_hat) / np.linalg.norm(np.cross(z_hat, h_hat))
    n_Omega_perp_hat = np.cross(h_hat, n_Omega_hat)
    Omega = np.arctan2(np.dot(n_Omega_hat, y_hat), np.dot(n_Omega_hat, x_hat))

    e = np.cross(v0, h) / mu - normalize(r0)
    e_hat = normalize(e)
    e_mag = np.linalg.norm(e)
    e_perp_hat = np.cross(h_hat, e_hat)
    omega = np.arctan2(np.dot(e_hat, n_Omega_perp_hat), np.dot(e_hat, n_Omega_hat))
    a = p / (1 - np.dot(e, e))

    f = np.arctan2(np.dot(r0,e_perp_hat),np.dot(r0,e_hat))

    tanE2 = np.sqrt((1.-e_mag)/(1.+e_mag))*np.tan(f/2.)
    E = 2*np.arctan2(tanE2,1)

    tau = t0 - np.sqrt(a**3/mu)*(E-e_mag*np.sin(E))

    OE = [a, e, i, omega, Omega, tau]

    return OE



