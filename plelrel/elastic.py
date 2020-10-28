import numpy
from scipy.optimize import fsolve

class sr_elasticity(object):
    """
    No source
    """
    
    def __init__(self, initial_data, gamma = 5/3):
        self.gamma = gamma
        self.Nvars = 13
        self.Nprim = 13
        self.Naux = 30
        self.Nadvect = 6
        self.initial_data = initial_data
        self.prim_names = (r"$\psi^X_x$", r"$\psi^X_y$", r"$\psi^X_z$", 
                           r"$\psi^Y_x$", r"$\psi^Y_y$", r"$\psi^Y_z$", 
                           r"$\psi^Z_x$", r"$\psi^Z_y$", r"$\psi^Z_z$", 
                           r"$v^x$", r"$v^y$", r"$v^z$", 
                           r"$\epsilon$")
        self.cons_names = (r"$\psi^X_x$", r"$\psi^X_y$", r"$\psi^X_z$", 
                           r"$\psi^Y_x$", r"$\psi^Y_y$", r"$\psi^Y_z$", 
                           r"$\psi^Z_x$", r"$\psi^Z_y$", r"$\psi^Z_z$", 
                           r"$S_x$", r"$S_y$", r"$S_z$", 
                           r"$\tau$")
        self.aux_names = (r"$\rho$", r"$p$", r"$W$", r"$h$", r"$c_s$", 
                          r"$p_{tt}$", r"$p_{tx}$", r"$p_{ty}$", r"$p_{tz}$", 
                                       r"$p_{xx}$", r"$p_{xy}$", r"$p_{xz}$",
                                                    r"$p_{yy}$", r"$p_{yz}$",
                                                                 r"$p_{zz}$",
                          r"$\eta_{XX}$", r"$\eta_{XY}$", r"$\eta_{XZ}$",
                                          r"$\eta_{YY}$", r"$\eta_{YZ}$",
                                                          r"$\eta_{ZZ}$",
                          r"$tr(\eta)$", r"$tr(\eta^2)$", r"$tr(\eta^3)$",
                          r"$\psi^X_t$", r"$\psi^Y_t$", r"$\psi^Z_t$")
        self.advect_names = (r"$k_{XX}$", r"$k_{XY}$", r"$k_{XZ}$",
                                          r"$k_{YY}$", r"$k_{YZ}$",
                                                       r"$k_{ZZ}$")
    
    
    def prim2all_point(self, prim, advected):
        # These are the primitive quantities we store
        psi = numpy.zeros((3, 4))
        psi[:, 1:] = numpy.reshape(prim[0:9], (3, 3))
        v_up = prim[9:12]
        epsilon = prim[12]
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
        # Now compute EOS etc
        # This uses the Toy_1 EOS
        f_1 = self.alpha * numpy.abs(I_1 - 3)
        f_2 = self.beta * numpy.abs(I_2 - 3)
        # Now compute pi
        pi_X_down = 2 * n * (f_1 * (eta_X_down - g_X_down * I_1 / 3) +
                             2 * f_2 * (eta_X_down @ eta_X_mixed -
                                        g_X_down * I_2 / 3))
        pi_M_down = psi.T @ pi_X_down @ psi
        # Construct full pressure tensor
        # This assumes we have p, in that it came from the primitives...
        p_M_down = p * h_M_down + pi_M_down
        
        
    
    def prim2all(self, prim):
        
    def cons_fn(self, guess, ...):
    
    def cons2all(self, cons, prim_old):
        return prim, aux
        
    def flux(self, cons, prim, aux):
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
