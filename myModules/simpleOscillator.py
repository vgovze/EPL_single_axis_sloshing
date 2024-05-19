import numpy as np

# The solutions to the of a dampened oscillator:
#
# m x'' + d x' + k x = F_0 sin(w t)
#
# in dimensionless form:
#
# u'' + b u' + c u = a0 sin(omega tau)
#
# Dimensionless numbers:
#
# b = d T / m          - dampening number
# c = k T^2 / m        - rigidity number
# a0 = F_0 T^2 / m / L - force number
# omega = w T          - wave number
# tau = t / T          - dimensionless time
# u = x / L            - dimensionless displacement
# u' = x' T / L        - dimensionless velocity
#
# Physical quantities:
#
# m - mass
# d - dampening coefficient
# k - stiffness coefficient
# F_0 - amplitude of the vibration
# t - time
# w - angular velocity
# x - displacement
# T - reference time
# L - reference length

# The unit conversion object.
class unitConverter:
    def to_b(self, d, T, m):
        return d*T/m
    def to_c(self, k, T, m):
        return k*T**2/m
    def to_a0(self, f0, T, m, L):
        return f0*T**2/m/L
    def to_omega(self, w, T):
        return w*T
    def to_tau(self, t, T):
        return t/T
    def to_u(self, x, L):
        return x/L
    def to_du(self, dx, T, L):
        return dx*T/L
    def to_x(self, u, L):
        return u*L
    def to_dx(self, du, T, L):
        return du*L/T
    def to_t(self, tau, T):
        return tau*T
    def to_d(self, b, T, m):
        return b*m/T
    def to_k(self, c, T, m):
        return c*m/T**2
    def to_f0(self, a0, T, m, L):
        return a0/T**2*m*L
    def to_w(self, omega, T):
        return omega/T

# The solutions to the governing equation.
def _discriminant(b, c):
    return b**2-4*c

# The interface for the simple oscillator solution.
class simpleOscillatorSol:
    def _u1(self, tau):
        return 0
    def _du1(self, tau):
        return 0
    def _u2(self, tau):
        return 0
    def _du2(self, tau):
        return 0
    def _up(self, tau):
        return 0
    def _dup(self, tau):
        return 0

    def amp(self, freqRat):
        res = np.sqrt(self.c**2*(1-freqRat**2)**2 \
                      + self.b**2*self.c*freqRat**2)
        return self.a0/res

    def freqRat(self):
        return self.omega**2/self.c

    def setParams(self, b, c, omega, a0):
        self.b = b
        self.c = c
        self.omega = omega
        self.a0 = a0

    def setInitialConditions(self, u0, du0):
        A = np.array([[self._u1(0),   self._u2(0)],\
                      [self._du1(0), self._du2(0)]])
        b = np.array([u0 - self._up(0), du0 - self._dup(0)])
        self.C1, self.C2 = np.linalg.solve(A, b)

    def u(self, tau):
        return self.C1*self._u1(tau)+self.C2*self._u2(tau)+self._up(tau)

    def du(self, tau):
        return self.C1*self._du1(tau)+self.C2*self._du2(tau)+self._dup(tau)

class _overdampedSol(simpleOscillatorSol):
    def _coef1(self, b, c):
        return 0.5*(-b+np.sqrt(_discriminant(b, c)))

    def _coef2(self, b, c):
        return 0.5*(-b-np.sqrt(_discriminant(b, c)))

    def _u1(self, tau):
        coef = self._coef1(self.b, self.c)
        return np.exp(coef*tau)

    def _du1(self, tau):
        coef = self._coef1(self.b, self.c)
        return coef*np.exp(coef*tau)

    def _u2(self, tau):
        coef = self._coef2(self.b, self.c)
        return np.exp(coef*tau)

    def _du2(self, tau):
        coef = self._coef2(self.b, self.c)
        return coef*np.exp(coef*tau)

    def _up(self, tau):
        coef = self.a0/(self.b**2*self.omega**2 + self.c**2 \
                        + self.omega**4 - 2*self.c*self.omega**2)
        return coef*(np.sin(self.omega*tau)*(self.c - self.omega**2) \
                     - np.cos(self.omega*tau)*self.b*self.omega)
 
    def _dup(self, tau):
        coef = self.a0*self.omega/(self.b**2*self.omega**2 + self.c**2 \
                                   + self.omega**4 - 2*self.c*self.omega**2)
        return coef*(np.cos(self.omega*tau)*(self.c - self.omega**2) \
                     + np.sin(self.omega*tau)*self.b*self.omega)

class _underdampedSol(_overdampedSol):

    def _coef1(self, b, c):
        return 0.5*np.sqrt(-_discriminant(b, c))

    def _coef2(self, b):
        return 0.5*b

    def _u1(self, tau):
        coef1 = self._coef1(self.b, self.c)
        coef2 = self._coef2(self.b)
        return np.exp(-coef2*tau)*np.cos(coef1*tau)

    def _du1(self, tau):
        coef1 = self._coef1(self.b, self.c)
        coef2 = self._coef2(self.b)
        return -np.exp(-coef2*tau)*(coef2*np.cos(coef1*tau) \
                                    + coef1*np.sin(coef1*tau))

    def _u2(self, tau):
        coef1 = self._coef1(self.b, self.c)
        coef2 = self._coef2(self.b)
        return np.exp(-coef2*tau)*np.sin(coef1*tau)

    def _du2(self, tau):
        coef1 = self._coef1(self.b, self.c)
        coef2 = self._coef2(self.b)
        return -np.exp(-coef2*tau)*(coef2*np.sin(coef1*tau) \
                                    - coef1*np.cos(coef1*tau))

class _critDampedSol(simpleOscillatorSol):

    def _coef(self, b):
        return -0.5*b

    def _u1(self, tau):
        coef = self._coef(self.b)
        return np.exp(coef*tau)

    def _du1(self, tau):
        coef = self._coef(self.b)
        return coef*np.exp(coef*tau)

    def _u2(self, tau):
        coef = self._coef(self.b)
        return tau*np.exp(coef*tau)

    def _du2(self, tau):
        coef = self._coef(self.b)
        return coef*np.exp(coef*tau)*tau + np.exp(coef*tau)

    def _up(self, tau):
        coef = 4*self.a0/(self.b**2 + 4*self.omega**2)**2
        return coef*(np.sin(self.omega*tau)*(self.b**2 - 4*self.omega**2) \
                     - np.cos(self.omega*tau)*4*self.b*self.omega)

    def _dup(self, tau):
        coef = 4*self.a0*self.omega/(self.b**2 + 4*self.omega**2)**2
        return coef*(np.cos(self.omega*tau)*(self.b**2 - 4*self.omega**2) \
                     + np.sin(self.omega*tau)*4*self.b*self.omega)

_sol = [_underdampedSol(), _critDampedSol(), _overdampedSol()]
_eps = np.finfo(float).eps

def getSolution(b, c) -> simpleOscillatorSol:
    D = _discriminant(b, c)
    if D > _eps:
        sol = _sol[2]
    elif D < -_eps:
        sol = _sol[0]
    else:
        sol = _sol[1]
    return sol