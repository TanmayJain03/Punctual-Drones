## IMPORTANT CONCEPT IDENTIFIED
# Set initial values at rest -> Desired x, y, z, u, v, w : -> compute 4 U's are desired input-> get desired w from them -> desired voltage from desired w -> motor model -> actual w (after sening)-> actual control input(from actual sensed w) (i dont think it canbe measerd though)-> model update law -> sense current position and attitude and compute error from sensors and complete the closed loop

## NOTE
# This is an ideal Backstepping controller + Plant model for Quadcopter with no disturbance and noise along with ideal actuators and sensors

## IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin

# TODO : 
# 1. Motor model
# 2. Time Varying Desired Trajectory
# 3. Adding Disturbance and Noise to model to see performance

# Next steps: 
# TODO: Add adaptivity and Robustness to current model

## INPUT [FOR MODEL]: 
# Initial Desired Traj: x, v, a ((x,y,z) & (r,p,y)) : 18 total
# Inital State: x, v ((x,y,z) & (r,p,y)): 12 total 

## OUTPUT [FOR MODEL]:
# Tracked states in discrete time steps: x, v ((x,y,z) & (r,p,y)): 12 total  

class Backstepping(): 
    def __init__(self, X_desired, X_inititial): # initial desired reference and initial state

        ## Abolute constants
        self.G = 9.8
        self.Mass = 0.4784
        self.Lwings = 0.225 #(COM to motor center)
        self.Ix = 0.0086
        self.Iy = 0.0086
        self.Iz = 0.0172
        self.Ir = 3.74*10**(-5) #rotor inertia
        self.h = 0.001

        ## Variable parameters
        self.Kp = 1 # lift coeff
        self.Kd = 0.025 # drag coeff (rotational)
        self.Kfax = 0 # Aerodynamic friction coeefs to calculate aerodynamic friction torques 
        self.Kfay = 0
        self.Kfaz = 0
        self.Kftx = 0 # Translation drag coeffs
        self.Kfty = 0
        self.Kftz = 0 

        self.alpha = np.ones((13), dtype='float64') * 1

        ## DERIVED CONSTANTS
        self.a1 = (self.Iy - self.Iz) / self.Ix
        self.a2 = -self.Kfax / self.Ix
        self.a3 = self.Ir / self.Ix
        self.a4 = (self.Iz - self.Ix) / self.Iy
        self.a5 = -self.Kfay / self.Iy
        self.a6 = self.Ir / self.Iy
        self.a7 = (self.Ix - self.Iy) / self.Iz
        self.a8 = -self.Kfaz / self.Iz
        self.a9 = -self.Kftx / self.Ix
        self.a10 = -self.Kfty / self.Iy
        self.a11 = -self.Kftz / self.Iz

        self.b1 = self.Lwings / self.Ix
        self.b2 = self.Lwings / self.Iy
        self.b3 = self.Lwings / self.Iz

        
        ##### STATES
        ## DESIRED
        self.X_des = np.copy(X_desired)      #np.zeros((7, 3), dtype='float64') # 6 states x,y,z,th, phi, psi ,[x3]:pos,vel,acc its der (velocity), and its double der (acc)

        ## CURRENT STATE
        self.X_curr = np.copy(X_inititial)

        ## NEXT STATE
        self.X_next = np.zeros((13), dtype='float64') 

        #### CONTROL INPUTS
        ## STATE INPUTS
        self.U = np.zeros((5), dtype='float64')
        self.Ux = cos(self.X_curr[1])*sin(self.X_curr[3])*cos(self.X_curr[5]) + sin(self.X_curr[1])*sin(self.X_curr[5])
        self.Uy = cos(self.X_curr[1])*sin(self.X_curr[3])*sin(self.X_curr[5]) - sin(self.X_curr[1])*cos(self.X_curr[5])

        ## MOTOR INPUTS
        self.Omega = np.zeros((5), dtype='float64')
        self.Omega_bar = self.Omega[1] - self.Omega[2] + self.Omega[3] - self.Omega[4]


    ## STATE UPDATE LAWS
    def phi_update(self):
        self.X_next[1] = self.X_curr[1] + self.h*(self.X_curr[2])
        self.X_next[2] = self.X_curr[2] + self.h*(self.a1*self.X_curr[4]*self.X_curr[6] + self.a2*self.X_curr[2]**2 +  self.a3*self.X_curr[4]*self.Omega_bar + self.b1*self.U[2] )

    def theta_update(self):
        self.X_next[3] = self.X_curr[3] + self.h*(self.X_curr[4])
        self.X_next[4] = self.X_curr[4] + self.h*(self.a4*self.X_curr[2]*self.X_curr[6] + self.a5*self.X_curr[4]**2 +  self.a6*self.X_curr[2]*self.Omega_bar + self.b2*self.U[3] )

    def psi_update(self):
        self.X_next[5] = self.X_curr[5] + self.h*(self.X_curr[6])
        self.X_next[6] = self.X_curr[6] + self.h*(self.a7*self.X_curr[2]*self.X_curr[4] + self.a8*self.X_curr[6]**2 + self.b3*self.U[4] )

    def x_update(self):
        self.X_next[7] = self.X_curr[7] + self.h*self.X_curr[8]
        self.X_next[8] = self.X_curr[8] + self.h*(self.a9*self.X_curr[8] + self.Ux*self.U[1]/self.Mass)


    def y_update(self):
        self.X_next[9] = self.X_curr[9] + self.h*self.X_curr[10]
        self.X_next[10] = self.X_curr[10] + self.h*(self.a10*self.X_curr[10] + self.Uy*self.U[1]/self.Mass)

    def z_update(self):
        self.X_next[11] = self.X_curr[11] + self.h*self.X_curr[12]
        self.X_next[12] = self.X_curr[12] + self.h*(self.a11*self.X_curr[12] + cos(self.X_curr[1])*cos(self.X_curr[3])*self.U[1]/self.Mass - self.G)


    ## CONTROL INPUTS UPDATE
    def u_update(self):

        ## 1 & 2(roll) 1st state X_des[1], alpha[1,2], U2, x1,2
        z1 = self.X_des[1][0] - self.X_curr[1]
        z2 = self.X_curr[2] - self.X_des[1][1] - self.alpha[1]*z1
        self.U[2] = 1/self.b1*(-self.a1*self.X_curr[4]*self.X_curr[6] - self.a2*self.X_curr[2]**2 - self.a3*self.Omega_bar*self.X_curr[4] + self.X_des[1][2] + self.alpha[1]*(self.X_des[1][1] - self.X_curr[2]) - self.alpha[2]*z2 + z1)

        ## 3 & 4 (pitch) 2nd State
        z1 = self.X_des[2][0] - self.X_curr[3]
        z2 = self.X_curr[4] - self.X_des[2][1] - self.alpha[3]*z1
        self.U[3] = 1/self.b2*(-self.a4*self.X_curr[2]*self.X_curr[6] - self.a5*self.X_curr[4]**2 - self.a6*self.Omega_bar*self.X_curr[2] + self.X_des[2][2] + self.alpha[3]*(self.X_des[2][1] - self.X_curr[4]) - self.alpha[4]*z2 + z1)

        ## 5 & 6 (yaw) 3rd State
        z1 = self.X_des[3][0] - self.X_curr[5]
        z2 = self.X_curr[6] - self.X_des[3][1] - self.alpha[5]*z1
        self.U[4] = 1/self.b3*(-self.a7*self.X_curr[2]*self.X_curr[4] - self.a8*self.X_curr[6]**2  + self.X_des[3][2] + self.alpha[5]*(self.X_des[3][1] - self.X_curr[6]) - self.alpha[6]*z2 + z1)

        ## 7 & 8 (x) 4th state
        z1 = self.X_des[4][0] - self.X_curr[7]
        z2 = self.X_curr[8] - self.X_des[4][1] - self.alpha[7]*z1
        if(self.U[1] != 0):
            self.Ux = self.Mass/self.U[1]*(-self.a9*self.X_curr[8] + self.X_des[4][2] + self.a7*(self.X_des[4][1] - self.X_curr[8]) - self.alpha[8]*z2 + z1)

        ## 9 & 10 (y) 5th state
        z1 = self.X_des[5][0] - self.X_curr[9]
        z2 = self.X_curr[10] - self.X_des[5][1] - self.alpha[9]*z1
        if(self.U[1] != 0):
            self.Uy = self.Mass/self.U[1]*(-self.a10*self.X_curr[10] + self.X_des[5][2] + self.a9*(self.X_des[5][1] -self.X_curr[10]) - self.alpha[10]*z2 + z1)

        ## 11 & 12 (z) 6th State
        z1 = self.X_des[6][0] - self.X_curr[11]
        z2 = self.X_curr[12] - self.X_des[6][1] - self.alpha[11]*z1
        self.U[1] = self.Mass/(cos(self.X_curr[1])*cos(self.X_curr[3]))*(self.G - self.a11*self.X_curr[12] + self.alpha[11]*(self.X_des[6][1] - self.X_curr[12]) - self.alpha[12]*z2 + z1)


    ## MOTOR INPUTS UPDATE
    def w_update(self):
        self.Omega[1] = np.sqrt(1/(4*self.Kp)*self.U[1] + 1/(2*self.Kp)*self.U[3] + 1/(4*self.Kd)*self.U[4])
        self.Omega[2] = np.sqrt(1/(4*self.Kp)*self.U[1] - 1/(2*self.Kp)*self.U[2] - 1/(4*self.Kd)*self.U[4])
        self.Omega[3] = np.sqrt(1/(4*self.Kp)*self.U[1] - 1/(2*self.Kp)*self.U[3] + 1/(4*self.Kd)*self.U[4])
        self.Omega[4] = np.sqrt(1/(4*self.Kp)*self.U[1] + 1/(2*self.Kp)*self.U[2] - 1/(4*self.Kd)*self.U[4])
        self.Omega_bar = self.Omega[1] - self.Omega[2] + self.Omega[3] - self.Omega[4]
    
    def next_step(self):
        self.X_curr = np.copy(self.X_next)

    def UpdateBacksteppingModel(self):
        self.phi_update()
        self.theta_update()
        self.psi_update()

        self.z_update()
        self.y_update()
        self.x_update()

        self.u_update()
        self.w_update()

        self.next_step()


def SetInitialConditions():
    X_ref = np.zeros((7, 3), dtype='float64')
    X_init = np.zeros((13), dtype='float64') 
    X_ref[4][0], X_ref[5][0], X_ref[6][0] = input("X, Y, Z: ").split()
    # X_init[11] = 1
    backstepping = Backstepping(X_ref, X_init)
    return backstepping

def PlotTrajectory(X, Y, Z):

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(X, Y, Z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':

    model = SetInitialConditions()
    
    T = []
    X_ser = []
    Y_ser = []
    Z_ser = []
    t = 100000
    for i in range(0, t):
        model.UpdateBacksteppingModel()
        T.append(i)
        X_ser.append(model.X_curr[7])
        Y_ser.append(model.X_curr[9])
        Z_ser.append(model.X_curr[11])

    PlotTrajectory(X_ser, Y_ser, Z_ser)