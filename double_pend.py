#Solves and animates the double pendulum for masses m1, m2, string lengths L1, L2, and given initial conditions for the 
#angles and angular velocities for the two masses. Equations of motions found from Lagrange equations, and solved numerically.
#Change u_0 to see animation for different initial values. Also calculates energy of pendulum, to check the solution.
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
from scipy import integrate
L1, L2 = 25,15 
m1,m2 = 1,1
g = 9.81
T = 100
dt = 0.01
a = 2 #determines number of frames: frames = t_steps / a

u_0 = [3, 0, 1, 0] # initial vals for the pendulum. with u_0 = [theta1_0, omega1_0, theta2_0, omega2_0]

def equation(t, u): #right side of diff.eq. left side: d^2(u)/dt^2, with u = [theta1, omega1, theta2, omega2]
    c, s = np.cos(u[0]-u[2]), np.sin(u[0]-u[2])
    dtheta1 = u[1]
    domega1 = (m2*g*np.sin(u[2])*c - m2*s*(L1*u[1]**2*c + L2*u[3]**2) -
             (m1+m2)*g*np.sin(u[0])) / L1 / (m1 + m2*s**2)
    dtheta2 = u[3]
    domega2 = ((m1+m2)*(L1*u[1]**2*s - g*np.sin(u[2]) + g*np.sin(u[0])*c) + 
             m2*L2*u[3]**2*s*c) / L2 / (m1 + m2*s**2)

    return [dtheta1, domega1, dtheta2, domega2]

def solve_equation(RHS, init_u, T, dt): #solves the diff.eq. returns u, t. uses Radau method, since this is a stiff problem
    t_span = np.array((0, T))
    t_arr = np.arange(0, T, dt)
    solution = integrate.solve_ivp(RHS, t_span, init_u, method = "Radau", t_eval = t_arr)
    theta1 = solution.y[0]
    theta2 = solution.y[2]
    omega1 = solution.y[1]   
    omega2 = solution.y[3]
    t = solution.t
    return theta1, theta2, omega1, omega2,  t

def energy(theta1, theta2, omega1, omega2): #calculates energy of pendulum
    V = - (m1 + m2) * L1 * g * np.cos(theta1) - m2 * L2 * g * np.cos(theta2)
    T = 0.5 * m1 * (L1 * omega1)**2 + 0.5 * m2 * (( L1 * omega1 )**2 + (L2 * omega2)**2 + 2 * L1 * L2 
            * omega1 * omega2 * np.cos(theta1-theta2))
    return T + V

def init(): #must use name init for funcAnimation. Initiates lines and circles.
    line1.set_data([], [])
    line2.set_data([], [])
    circle1.center = (0,0)
    circle1.center = (0,0)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    return line1, line2, circle1, circle2

def animate(i): # amination-function. i = frame number. 
    x1, y1 = L1 * np.sin( theta1[i] ), - L1* np.cos(theta1[i] ) 
    x2, y2 = x1 + L2 * np.sin( theta2[i] ),  y1 - L2* np.cos(theta2[i] ) 
    circle1.center, circle2.center = (x1, y1), (x2, y2)
    xdata1[i-1], ydata1[i-1] = x1, y1
    xdata1[i], ydata1[i] = x2, y2
    xdata1[i-2], ydata1[i-2] = 0, 0
    xdata2[i], ydata2[i] = x2, y2
    line1.set_data(xdata1[i-2:i+1], ydata1[i-2:i+1]) 
    line2.set_data(xdata2[:i+1], ydata2[:i+1])
    return line1, line2 , circle1, circle2

theta1, theta2, omega1, omega2,  t = solve_equation(equation, u_0 , T , dt )

theta1 = theta1[::a] #animate only every a'th value 
theta2 = theta2[::a]
omega1 = omega1[::a]
omega2 = omega2[::a]
t = t[::a]

circle1, circle2 = plt.Circle((0, 0), 0.5, color='r'), plt.Circle((0, 0), 0.5, color='r') #make circle obj center in (0,0), r = radius
plt.style.use('dark_background') 
fig = plt.figure( figsize = (8, 8))
ax = plt.axes(xlim = (-50, 50), ylim = (-50 , 50))
line1, = ax.plot([], [], lw = 2, color = "c")
line2, = ax.plot([], [], lw = 1, color ="r" )
frames =  int( T / dt / a ) 
xdata1, ydata1, xdata2, ydata2  = np.zeros(frames), np.zeros(frames), np.zeros(frames), np.zeros(frames)  #reason for (x,y) = (0,0) always drawn
 
plt.title("Double pendulum")
plt.axis("off") 
#animate
anim = animation.FuncAnimation(fig, animate, init_func= init, frames = frames, interval = dt  ) #interval given in ms
plt.draw()
plt.show()

#plot energy of system
E = energy(theta1, theta2, omega1, omega2)
plt.plot(t, E )
plt.ylim( np.amin(E) - abs(np.amin(E)), np.amax(E) + abs(np.amax(E)) )
plt.title("Energy of the pendulum")
plt.ylabel("Energy[J]")
plt.xlabel("Time[s]")
plt.show()
