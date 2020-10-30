import numpy
from scipy.optimize import fsolve

class elastic_eos_toy(object):
    
    def __init__(eos_gamma, eos_lambda, eos_kappa):
        self.eos_gamma = eos_gamma
        self.eos_lambda = eos_lambda
        self.eos_kappa = eos_kappa
        
    def enthalpy(n, entropy, shear_S):
        return (1 + (entropy * n**(self.eos_gamma - 1) 
                     * (self.eos_gamma / (self.eos_gamma - 1))) + 
                self.eos_lambda * self.eos_kappa * n**(self.eos_lambda - 1) * shear_S)
    
    def pressure(n, entropy, shear_S):
        return (entropy * n**(self.eos_gamma) + 
             (self.eos_lambda - 1) * self.eos_kappa * n**(self.eos_lambda) * shear_S)

    def fs(n, entropy, shear_S, I_1, I_2):
        f_1 = (self.eos_kappa * n**(self.eos_lambda - 1) * 
               (3 * I_1**2 - I_2) / 24)
        f_2 = -self.eos_kappa * n**(self.eos_lambda - 1) * I_1 / 24
        return f_1, f_2
    

class sr_elastic_point(object):
    """
    Does the conversion between the different forms at a point
    """
    
    def __init__(self, eos, prim, advected):
        # Store the eos
        self.eos = eos
        # These are the primitive quantities we store
        psi = numpy.zeros((3, 4))
        psi[:, 1:] = numpy.reshape(prim[0:9], (3, 3))
        v_up = prim[9:12]
        entropy = prim[12]
        # These are the advected quantities, which are the reference metric:
        k_X_down = numpy.reshape(advected[0:9], (3, 3))
        # The map has timelike components, which need computing:
        psi[:, 0] = numpy.dot(psi[:, 1:], v_up)
        # The Lorentz factor is simple in flat space
        W = 1 / numpy.sqrt(1 - numpy.dot(v_up, v_up))
        # The four velocity follows
        u_up = W * numpy.array([-1.0, v_up[0], v_up[1], v_up[2]])
        # The Minkowski metric gives us the projector
        g_M_up = numpy.diag([-1.0, 1.0, 1.0, 1.0])
        h_M_up = g_M_up + numpy.outer(u_up, u_up)
        g_M_down = g_M_up # Minkowski!
        h_M_down = g_M_down @ h_M_up @ g_M_down.T
        # Get the other velocities
        u_down = g_M_down @ u_up
        v_down = g_M_down[1:, 1:] @ v_up
        # Now project the matrix to the reference space
        g_X_up = psi @ g_M_up @ psi.T
        # Now take the advected metric and raise an index
        k_X_mixed = g_X_up @ k_X_down
        # The number density is the square root of the determinant of this
        n = numpy.sqrt(numpy.det(k_X_mixed))
        # Now we need the inverse of g on the reference space
        g_X_down = numpy.inv(g_X_up)
        # And then we need eta to compute invariants:
        eta_X_down = k_X_down / n**(2/3)
        eta_X_mixed = k_X_mixed / n**(2/3)
        # Now compute invariants
        I_1 = numpy.trace(eta_X_mixed)
        I_2 = numpy.trace(eta_X_mixed @ eta_X_mixed)
        # Now the shear scalar, toy EOS style
        shear_S = (I_1**3 - I_1 * I_2 - 18) / 24
        # Now compute EOS etc
        # This is the Toy_2 EOS, which is (I.4-10) in GHE.
        # The entropy that's stored in the primitive variable is used as K(s)
        enthalpy = eos.enthalpy(n, entropy, shear_S)
        p = eos.pressure(n, entropy, shear_S)
        epsilon = enthalpy - 1 - p / n
        # This uses the Toy_2 EOS
        f_1, f_2 = eos.fs(n, entropy, shear_S, I_1, I_2)
        # Now compute pi
        pi_X_down = 2 * n * (f_1 * (eta_X_down - g_X_down * I_1 / 3) +
                             2 * f_2 * (eta_X_down @ eta_X_mixed -
                                        g_X_down * I_2 / 3))
        pi_M_down = psi.T @ pi_X_down @ psi
        # Construct full pressure tensor
        p_M_down = p * h_M_down + pi_M_down
        # Now construct the von Mises scalars
        pi_X_mixed = g_X_up @ pi_X_down
        J_1 = numpy.trace(pi_X_mixed)
        J_2 = numpy.trace(pi_X_mixed @ pi_X_mixed)
        # Now construct the conserved variables
        # Have taken advantage of Minkowski space in many places here.
        S = n * enthalpy * W**2 * v_down + (g_M_up @ pi_M_down)[0, 1:]
        tau = n * (enthalpy * W**2 - W) - p + pi_M_down[0, 0]
        # Now construct the fluxes at this point
        # This is solely the flux in the x direction
        f_psi = numpy.zeros((3, 3))
        f_psi[:, 0] = psi @ v_up
        f_S = (n * enthalpy * W**2 * v_up[0] * v_down +
               (g_X_up @ p_M_down)[1, 1:])
        f_tau = (n * (enthalpy * W**2 - W) * v_up[0] +
                 numpy.dot((g_M_up @ pi_M_down)[1, 1:], v_up))
        # Now store everything
        self.psi = psi
        self.v_up = v_up
        self.W = W
        self.n = n
        self.p = p
        self.epsilon = epsilon
        self.entropy = entropy
        self.h = h
        self.k_X_down = k_X_down
        self.k_X_mixed = k_X_mixed
        self.g_X_up = g_X_up
        self.g_X_down = g_X_down
        self.g_M_up = g_M_up
        self.g_M_down = g_M_down
        self.eta_X_down = eta_X_down
        self.eta_X_up = eta_X_up
        self.I_1 = I_1
        self.I_2 = I_2
        self.J_1 = J_2
        self.J_2 = J_2
        self.f_1 = f_1
        self.f_2 = f_2
        self.pi_X_down = pi_X_down
        self.pi_X_mixed = pi_X_mixed
        self.pi_M_down = pi_M_down
        self.p_M_down = p_M_down
        self.S = S
        self.tau = tau
        self.f_psi = f_psi
        self.f_S = f_S
        self.f_tau = f_tau
        
    def fluxes(self):
        f = numpy.zeros((13,))
        f[:9] = self.f_psi.ravel()
        f[9:12] = self.f_S
        f[12] = self.f_tau
        return f
    
    def all_vars(self):
        prim = numpy.zeros((13,))
        cons = numpy.zeros((13,))
        auxl = numpy.zeros((28,))
        advected = numpy.zeros((6,))
        prim[:9] = self.psi[:, 1:].ravel()
        prim[9:12] = self.v_up
        prim[12] = self.entropy
        cons[:9] = prim[:9]
        cons[9:12] = self.S
        cons[12] = self.tau
        auxl[0] = self.n
        auxl[1] = self.epsilon
        auxl[2] = self.p
        auxl[3] = self.W
        auxl[4] = self.enthalpy
        auxl[5:9] = self.p_M_down[0, :]
        auxl[9:12] = self.p_M_down[1, 1:]
        auxl[12:14] = self.p_M_down[2, 2:]
        auxl[14:15] = self.p_M_down[3, 3:]
        auxl[15:18] = self.eta_X_down[0, :]
        auxl[18:20] = self.eta_X_down[1, 1:]
        auxl[20:21] = self.eta_X_down[2, 2:]
        auxl[21] = self.I_1
        auxl[22] = self.I_2
        auxl[23] = self.J_1
        auxl[24] = self.J_2
        auxl[25:28] = self.psi[:, 0]
        advected = self.k_X_down.ravel()
        return prim, cons, auxl, advected

class sr_elasticity(object):
    """
    No source
    """
    
    def __init__(self, initial_data,
                 eos_gamma = 5/3, eos_kappa = 1, eos_lambda = 1):
        self.eos = elastic_eos_toy(eos_gamma, eos_lambda, eos_kappa)
        self.gamma = gamma
        self.Nvars = 13
        self.Nprim = 13
        self.Naux = 28
        self.Nadvect = 6
        self.initial_data = initial_data
        self.prim_names = (r"$\psi^X_x$", r"$\psi^X_y$", r"$\psi^X_z$", 
                           r"$\psi^Y_x$", r"$\psi^Y_y$", r"$\psi^Y_z$", 
                           r"$\psi^Z_x$", r"$\psi^Z_y$", r"$\psi^Z_z$", 
                           r"$v^x$", r"$v^y$", r"$v^z$", 
                           r"$s$")
        self.cons_names = (r"$\psi^X_x$", r"$\psi^X_y$", r"$\psi^X_z$", 
                           r"$\psi^Y_x$", r"$\psi^Y_y$", r"$\psi^Y_z$", 
                           r"$\psi^Z_x$", r"$\psi^Z_y$", r"$\psi^Z_z$", 
                           r"$S_x$", r"$S_y$", r"$S_z$", 
                           r"$\tau$")
        self.aux_names = (r"$\rho$", r"$\epsilon$", r"$p$", r"$W$", r"$h$",
                          r"$p_{tt}$", r"$p_{tx}$", r"$p_{ty}$", r"$p_{tz}$", 
                                       r"$p_{xx}$", r"$p_{xy}$", r"$p_{xz}$",
                                                    r"$p_{yy}$", r"$p_{yz}$",
                                                                 r"$p_{zz}$",
                          r"$\eta_{XX}$", r"$\eta_{XY}$", r"$\eta_{XZ}$",
                                          r"$\eta_{YY}$", r"$\eta_{YZ}$",
                                                          r"$\eta_{ZZ}$",
                          r"$I_1$", r"$I_2$", r"$J_1$", r"$J_2$",
                          r"$\psi^X_t$", r"$\psi^Y_t$", r"$\psi^Z_t$")
        self.advect_names = (r"$k_{XX}$", r"$k_{XY}$", r"$k_{XZ}$",
                                          r"$k_{YY}$", r"$k_{YZ}$",
                                                       r"$k_{ZZ}$")
    
    
    def prim2all_point(self, prim, advected):
        
    
    def prim2all(self, prim):
        
    def cons_fn(self, guess, S, tau, psi, g_M_up, k_X_down):
        # Guesses are pi_{ik} v^k (3) and p - pi^{00} + D
        pi_v = guess[:3]
        p_pi_D = guess[3]
        v_down = (S + pi_v) / (tau + p_pi_D)
        v_up = v_down  # Minkowski
        W = 1 / np.sqrt(1 - np.dov(v_down, v_down))
        psi[:, 0] = -np.dot(psi[:, 1:], v_down)
        g_X_up = psi @ g_M_up @ psi.T
        k_X_mixed = g_X_up @ k_X_down
        n = np.det(k_X_mixed)**(1/2)
        eta_X_down = k_X_down / n**(2/3)
        eta_X_mixed = k_X_mixed / n**(2/3)
        # Now compute invariants
        I_1 = numpy.trace(eta_X_mixed)
        I_2 = numpy.trace(eta_X_mixed @ eta_X_mixed)
        # Now the shear scalar, toy EOS style
        shear_S = (I_1**3 - I_1 * I_2 - 18) / 24
        # Now compute EOS etc
        # This is the Toy_2 EOS, which is (I.4-10) in GHE.
        # The entropy that's stored in the primitive variable is used as K(s)
        enthalpy = eos.enthalpy(n, entropy, shear_S)
        p = eos.pressure(n, entropy, shear_S)
        epsilon = enthalpy - 1 - p / n
        # This uses the Toy_2 EOS
        f_1, f_2 = eos.fs(n, entropy, shear_S, I_1, I_2)
        # Now compute pi
        pi_X_down = 2 * n * (f_1 * (eta_X_down - g_X_down * I_1 / 3) +
                             2 * f_2 * (eta_X_down @ eta_X_mixed -
                                        g_X_down * I_2 / 3))
        pi_M_down = psi.T @ pi_X_down @ psi
        # Now we can compare to the guesses
        residual = numpy.zeros_like(guess)
        residual[:3] = pi_v - numpy.dot(pi_M_down[1:, 1:], v_up)
        residual[3] = p - pi_M_down[0, 0] - n * W
        
        return residual
        
        
    def cons2all_point(self, cons, advected, prim_old):
        psi = numpy.zeros((3, 4))
        psi[:, 1:] = numpy.reshape(cons[0:9], (3, 3))
        S = cons[9:12]
        tau = cons[12]
        k_X_down = numpy.reshape(advected, (3, 3))
        # Guesses are pi_{ik} v^k (3) and p - pi^{00} + D
        
        
    
    def cons2all(self, cons, advected, prim_old):
        
        return prim, aux
    
    def flux(self, cons, prim, aux, advected):
        return f
        
    def max_lambda(self, cons, prim, aux):
        """
        Laziness - speed of light
        """
        return 1
        
    def riemann_problem_flux(self, q_L, q_R):
        """
        Not implemented - cost
        """
        raise NotImplementedError('Exact Riemann solver is too expensive.')

def initial_riemann(ql, qr):
    return lambda x : numpy.where(x < 0.0,
                                  ql[:,numpy.newaxis]*numpy.ones((13,len(x))),
                                  qr[:,numpy.newaxis]*numpy.ones((13,len(x))))
