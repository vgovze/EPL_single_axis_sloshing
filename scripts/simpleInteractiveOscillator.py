import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, Button

from context import myModules, CONFIGPATH
from myModules.simpleOscillator import getSolution, unitConverter

# Set custom plotting style.
plt.style.use(os.path.join(CONFIGPATH, "interactive.mplstyle"))

# Instantiate the unit converter.
uConv = unitConverter()

# Set units of time and displacement:
T = 1 # second
L = 1 # metre

# The initial parameters.
m0 = 1
d0 = 1
k0 = 1
f0_0 = 1
w0 = 1
x0_0 = 1
dx0_0 = 1

# Initialize the variable objects.
class solution:
    def __init__(self, T, L):
        self.sol = None
        self.uConv = unitConverter()
        self.T = T
        self.L = L

    def update(self,m, d, k, f0, w, x0, dx0):
        b = self.uConv.to_b(d, self.T, m)
        c = self.uConv.to_c(k, self.T, m)
        a0 = self.uConv.to_a0(f0, self.T, m, self.L)
        omega = self.uConv.to_omega(w, self.T)
        u0 = self.uConv.to_u(x0, self.L)
        du0 = self.uConv.to_du(dx0, self.T, self.L)

        self.sol = getSolution(b, c)
        self.sol.setParams(b, c, omega, a0)
        self.sol.setInitialConditions(u0, du0)

    def getDisplacement(self, time):
        return self.uConv.to_x(self.sol.u(time), self.L)
    
    def getAmplitude(self, freqRat):
        return self.uConv.to_x(self.sol.amp(freqRat), self.L)

    def getFreqRat(self):
        return self.sol.freqRat()

# The time array.
time = np.linspace(0, 100, 1000)

# The frequency ratio.
freqRat = np.linspace(0, 2, 1000)

# The initial conditions.
x0 = x0_0
dx0 = dx0_0

sol = solution(T, L)
sol.update(m0, d0, k0, f0_0, w0, x0_0, dx0_0)

# Plot parameters.
cm = 1/2.54       # inches
figsizeX = 30*cm
figsizeY = 15*cm

# Set the canvas.
fig = plt.figure(figsize=(figsizeX, figsizeY))
timeAx = fig.add_subplot(1, 2, 1)
ampAx = fig.add_subplot(1, 2, 2)

# Set the time axes and plot.
timeAx.set(xlabel=r"$t$ / s", ylabel=r"$x$ / m",
           title="System dynamic response")
timePlt, = timeAx.plot(time, sol.getDisplacement(time), color="k")

# Set the omega axes and plot.
ampAx.set(xlabel=r"$\frac{\omega}{\omega_0}$", ylabel="$A$ / m",
            title="Steady-state amplitude plot")
ampPlt, = ampAx.plot(freqRat, sol.getAmplitude(freqRat), color="k")
vLinePlt, = ampAx.plot(
    [sol.getFreqRat(), sol.getFreqRat()],
    [0, sol.getAmplitude(sol.getFreqRat())], color="r",
    alpha=0.2, label = "Current"
)
ampAx.legend()

# Adjust the plot layer.
fig.subplots_adjust(right=0.9, wspace=0.25, top=0.9, bottom=0.25, left=0.2)

# Equation text.
equation = fig.text(
    x = 0.45,
    y = 0.95,
    s = r"$m \ddot{x}(t) + d \ddot{x}(t) + k x(t) = F_0 \sin(\omega t)$",
    fontsize=12
)

# Put the interactive GUI elements.
wAx = fig.add_axes([0.03, 0.25, 0.03, 0.65])
wSlider = Slider(
    ax=wAx,
    label='$\omega$ / rad/s',
    valmin=0,
    valmax=10,
    valinit=w0,
    valstep=0.1,
    orientation="vertical"
)

# Force amplitude.
f0Ax = fig.add_axes([0.1, 0.25, 0.03, 0.65])
f0Slider = Slider(
    ax=f0Ax,
    label='$F_0$ / N',
    valmin=0,
    valmax=1,
    valinit=f0_0,
    valstep=0.1,
    orientation="vertical"
)

x0Ax = fig.add_axes([0.08, 0.13, 0.05, 0.03])
x0Box = TextBox(
    ax=x0Ax,
    label=r"$x(t=0)$ / m",
    initial=f"{x0_0}",
    textalignment="center"
)

dx0Ax = fig.add_axes([0.08, 0.06, 0.05, 0.03])
dx0Box = TextBox(
    ax=dx0Ax,
    label=r"$\frac{dx}{dt}(t=0)$ / m/s",
    initial=f"{dx0_0}",
    textalignment="center"
)

# Mass.
mAx = fig.add_axes([0.2, 0.05, 0.7, 0.03])
mSlider = Slider(
    ax=mAx,
    label='$m$ / kg',
    valmin=0,
    valmax=10,
    valinit=m0,
    valstep=0.1,
)

# Damping coefficient.
dAx = fig.add_axes([0.2, 0.09, 0.7, 0.03])
dSlider = Slider(
    ax=dAx,
    label='$d$ / Ns/m',
    valmin=0,
    valmax=10,
    valinit=d0,
    valstep=0.1,
)

# Stiffness coefficient.
kAx = fig.add_axes([0.2, 0.13, 0.7, 0.03])
kSlider = Slider(
    ax=kAx,
    label='$k$ / N/m',
    valmin=0,
    valmax=10,
    valinit=d0,
    valstep=0.1,
)

# The update function.
def update(val):

    sol.update(
        mSlider.val,
        dSlider.val,
        kSlider.val,
        f0Slider.val,
        wSlider.val,
        x0,
        dx0
    )

    res = sol.getDisplacement(time)
    timePlt.set_ydata(res)
    timeAx.set_ylim(1.1*min(res), 1.1*max(res)+0.001)

    res = sol.getAmplitude(freqRat)
    ampPlt.set_ydata(res)
    ampAx.set_ylim(0, 1.1*max(res)+0.0001)

    res = sol.getFreqRat()
    vLinePlt.set_xdata([res, res])
    vLinePlt.set_ydata([0, sol.getAmplitude(res)])

    fig.canvas.draw_idle()

# The reset button.
resetAx = fig.add_axes([0.91, 0.91, 0.08, 0.08])
resetButton = Button(
    ax=resetAx,
    label="Reset"
)

def reset(event):
    mSlider.reset()
    dSlider.reset()
    kSlider.reset()
    f0Slider.reset()
    wSlider.reset()
    x0Box.set_val(str(x0_0))
    dx0Box.set_val(str(dx0_0))

def evaluate_x0Box(expression):
    global x0
    try:
        val = float(eval(expression))
        x0Box.color = ".95"
        x0 = val
        update(1)
    except:
        x0Box.color = "red"
        return

def evaluate_dx0Box(expression):
    global dx0
    try:
        val = float(eval(expression))
        dx0Box.color = ".95"
        dx0 = val
        update(1)
    except:
        x0Box.color = "red"
        return

# What to do when something changes.
wSlider.on_changed(update)
mSlider.on_changed(update)
dSlider.on_changed(update)
kSlider.on_changed(update)
f0Slider.on_changed(update)
resetButton.on_clicked(reset)
x0Box.on_submit(evaluate_x0Box)
dx0Box.on_submit(evaluate_dx0Box)

plt.show()