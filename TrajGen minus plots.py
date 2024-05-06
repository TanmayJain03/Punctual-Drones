# TODO:
# 1. How do i define my nomimal trajectory (formulation done, passing remains)
# 2. MIP (done)
# 3. TARGET < ROBOTS
# 4. TARGET > ROBOTS
# NAME, LOOP IT TO CONTINUE FROM PREVIOUS STEP

import numpy as np
import matplotlib.pyplot as plt
import time
import cvxpy as cp
from scipy.optimize import milp
from scipy.optimize import LinearConstraint
from matplotlib.animation import FuncAnimation

class PARAMS:
    def __init__(self):
        self.R_COL = 0.2
        self.dt = 0.5
        self.TOTAL_TIME = 10
        self.number_robots = 15
        self.EPSILON_SCP = 0.1  *self.number_robots*(0.25*self.TOTAL_TIME/self.dt)
        self.INIT_POSE = np.array([(0, i, 2*i/5) for i in range(self.number_robots)])
        self.INIT_VEL = np.array([(0, 0, 0) for i in range(self.number_robots)])
        self.TARGETS = np.array([(1, i, 1) for i in range(self.number_robots, 0, -1)])
        self.TARGET_VEL = np.array([(0, 0, 0) for i in range(self.number_robots)])

        # Animation axes
        self.AxesLimits = [[-1, 4],  #X
                           [-2, 3],  #Y
                           [-1, 2]]  #Z
        self.letter = np.array([ (-2,-2,0),
    (-2,-1, 0),
    (-2, 0, 0),
    (-2, 1, 0),
    (-2, 2, 0),
    (-1.5, 1.5, 0),
    (-0.5, 0.5, 0),
    (0, 0, 0),
    (0.5, -0.5, 0),
    (1.5, -1.5, 0),
    (2, -2, 0),
    (2, -1, 0),
    (2, 0, 0),
    (2, 1, 0),
    (2, 2, 0)
    ]) 
        self.TARGETS = np.copy(self.letter )      

class Utils:
    def __init__(self, params):
        self.params = params

    def MatrixPower(self,A, n):
        if n ==0:
            return np.identity(len(A))
        if n ==1:
            return np.copy(A)
        An = np.copy(A)
        for i in range(n-1):
            An = np.matmul(An, A)
        return np.copy(An)
    
    def model(self, x_current, v_current, u_input):
        # X[k+1] = [1, h; 0, 1]X + [0; h]U
        x2_new = v_current + self.params.dt*u_input
        x1_new = x_current + self.params.dt*x2_new
        x_new = np.array(([x1_new, x2_new]))
        return (x_new[0], x_new[1])
    
    def animation(self, x_traj, y_traj, z_traj):             
        x_traj = np.array(x_traj)
        y_traj = np.array(y_traj)
        z_traj = np.array(z_traj)
        self.params.AxesLimits[0][0] = np.min(x_traj) -2
        self.params.AxesLimits[0][1] = np.max(x_traj) +2
        self.params.AxesLimits[1][0] = np.min(y_traj) -2
        self.params.AxesLimits[1][1] = np.max(y_traj) +2
        self.params.AxesLimits[2][0] = np.min(z_traj) -2
        self.params.AxesLimits[2][1] = np.max(z_traj) +2

        def random_walk(num_steps, index, max_step=0.05):
            """Return a 3D random walk as (num_steps, 3) array."""
            walk = np.array([x_traj[:, index], y_traj[:, index], z_traj[:, index]]).T
            return walk
        
        def update_lines(num, walks, lines):
            for line, walk in zip(lines, walks):
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_data(walk[:num, :2].T)
                line.set_3d_properties(walk[:num, 2])
            return lines

        # Data: 40 random walks as (num_steps, 3) arrays
        num_steps = len(z_traj)
        walks = [random_walk(num_steps, index) for index in range(self.params.number_robots)]

        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # Create lines initially without data
        lines = [ax.plot([], [], [])[0] for _ in walks]

        # Setting the axes properties
        ax.set(xlim3d=(self.params.AxesLimits[0][0], self.params.AxesLimits[0][1]), xlabel='X')
        ax.set(ylim3d=(self.params.AxesLimits[1][0], self.params.AxesLimits[1][1]), ylabel='Y')
        ax.set(zlim3d=(self.params.AxesLimits[2][0], self.params.AxesLimits[2][1]), zlabel='Z')

        # Creating the Animation object
        ani = FuncAnimation(
            fig, update_lines, num_steps*3, fargs=(walks, lines), interval=50)
        plt.show()


class assignment:
    def __init__(self, params, utils):
        self.params = params
        self.utils = utils

        self.TARGET_SET = None
        self.INIT_POSE = self.params.INIT_POSE
        self.INIT_VEL = self.params.INIT_VEL
        self.num_robots = len(self.INIT_POSE)

        self.TARGETS = self.params.TARGETS
        self.TARGET_VEL = self.params.TARGET_VEL
        self.num_targets = len(self.TARGETS)
        
        self.COST = np.zeros((self.num_robots, self.num_targets))
        self.get_cost()

        self.DIMENSION = 3
        self.STEPS = int(self.params.TOTAL_TIME/self.params.dt)

        self.OptimalAssignment = self.MILP()
        self.OptimalTargets = self.TARGETS                 #self.get_optimal_targets()
        
    def get_cost(self):
        for i in range(self.num_robots):
            for j in range(self.num_targets):
                self.COST[i][j] = np.linalg.norm(self.INIT_POSE[i] - self.TARGETS[j])

    def Auction_VSDAA(self):
        pass

    def MILP(self):
        # minimize cost @ X.T
        # wrt sum X[i,:] = 1, sum X[:,j] = 1
        # x_ij =0 or 1

        X = list(np.zeros((self.num_robots* self.num_targets, 1)))
        flat_cost = self.COST.flatten()

        def MIP_main():
            A = np.zeros((self.num_targets + self.num_robots, self.num_robots*self.num_targets), dtype='int8')
            for i in range(self.num_targets): # only for robot == target case
                for j in range(self.num_robots):
                    A[i][self.num_robots*j + i] = 1

            for i in range(self.num_targets, 2*self.num_targets): # only for robot == target case
                for j in range(self.num_robots):
                    A[i][self.num_robots*(i-self.num_targets) + j] = 1

            b_u = np.ones((self.num_robots*2, ))
            b_l = np.ones((self.num_robots*2, ))

            constraints = LinearConstraint(A, b_l, b_u)
            integrality = np.ones_like(flat_cost)

            result = milp(c=flat_cost, constraints=constraints, integrality=integrality)

            J_mip = np.zeros(self.num_robots, dtype='int8')
            for i in range(self.num_targets):
                for j in range(self.num_robots):
                    if(result.x[i*self.num_robots + j] == 1):
                        J_mip[i] = j
            return J_mip
            
        return MIP_main()
    
    def get_optimal_targets(self):
        target_i = []
        for i in range(self.num_robots):
            target_i.append(self.TARGETS[self.OptimalAssignment[i]])
        target_i = np.array(target_i)
        return target_i
    
    def get_nominal_trajectory_and_plots(self):

        x = self.INIT_POSE
        v = self.INIT_VEL
        Array = [(x,v)]
        X = [x]
        y_traj = [x[:,1]]
        x_traj = [x[:,0]]
        z_traj = [x[:,2]]
        u_nom = []
        
        for i in range(self.STEPS):
            k=1
            u = k*(-(x+v) + self.OptimalTargets) # lyapunov function control
            u_nom.append(u)
            Array.append(self.utils.model(x, v, u))
            x = Array[-1][0]
            v = Array[-1][1]
            X.append(Array[-1][0])
        
            y_traj.append(X[-1][:,1])
            x_traj.append(X[-1][:,0])
            z_traj.append(X[-1][:,2])
        
        u_nom.append(u*0)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(z_traj, y_traj, x_traj)
        plt.show()

        self.utils.animation(x_traj, y_traj, z_traj)
        return np.array(Array), np.array(u_nom)


class OptimalTrajectory:
    def __init__(self, X_nominal, U_nominal, assignment):
        self.utils = assignment.utils
        self.assignment = assignment
        self.params = assignment.params

        self.EPSILON_SCP = 0.1
        self.F = np.array([[1, self.params.dt],
                           [0, 1]])
        self.G = G = np.array([[0],
                               [self.params.dt]])
        self.iterations = self.assignment.STEPS + 1

        self.x_nominal = X_nominal
        self.u_nominal = U_nominal

        self.u_input = {}
        self.A = np.zeros((self.assignment.num_robots, self.assignment.DIMENSION*3, np.shape(U_nominal)[0]*np.shape(U_nominal)[2]))
        self.B = np.zeros((self.assignment.num_robots, self.assignment.DIMENSION*3))
        self.g = np.zeros((self.assignment.num_robots, (self.iterations-1)*self.assignment.num_robots*(self.assignment.num_robots-1)//2, np.shape(U_nominal)[0]*np.shape(U_nominal)[2]))  ######
        self.h = np.zeros((self.assignment.num_robots, (self.iterations-1)*self.assignment.num_robots*(self.assignment.num_robots-1)//2))

        self.OptimalTraj = None

    
    def obejective_function(self):
        # u = None # to optimize u.T @ u ; initialized to u_nominal
                # u = cp.Variable(np.shape(u_nom))
        # Create a 3D k-by-m-by-n variable.

        for i in range(self.assignment.num_robots):
            self.u_input[i] = cp.Variable((np.shape(self.u_nominal)[0]*np.shape(self.u_nominal)[2]))  #(101, 5, 2) u[robot_number] = (timestep, x/y)

    def equality_contraints(self):
        # [not a condition explicitly] - State transition ; X[k+1] = A @ X[k]  +  B @ u[k]  (assuming Time Invariance)
        # [not a condition explicitly] - Iniatial State ; For all  robots X[jth robot][0th time] = Xo[j] (inital_poses)
        # Final State; For all robots X[jth robot][Tth time] = Xf[j] (final_poses)
        # x_final[j] = A @ x_initial[j] + B @ sum( u[j] from 0 to T-1)
        # B-1(target[j][1/2] - MatrixPower(A, steps-1) @ init_poses) = CU[j] : Cx (1, 101) all ones excpet 101, similarly Cy
        # u_final = 0 , u[j][steps-1][1/2] = 0 (done)
        # Ax (1,101) Ay (1,101) | Ax[100], Ay[100] = 0
        
        for robot in range(self.assignment.num_robots):  # terminal u condition (At terminal time U_input must be zero)
            for k in range(0, self.assignment.DIMENSION):
                self.A[robot][k][k*self.iterations + self.iterations-1] = 1
                self.B[robot][k] = 0

        for robot in range(self.assignment.num_robots):
            for k in range(0, self.assignment.DIMENSION):
                for time in range(0, self.iterations-1):  
                    self.A[robot][self.assignment.DIMENSION + k][k*self.iterations + time] = np.matmul( self.utils.MatrixPower(self.F, self.assignment.STEPS - 1 - time), self.G)[0][0]
                    self.B[robot][self.assignment.DIMENSION + k] = (np.array([self.assignment.OptimalTargets[robot], self.assignment.TARGET_VEL[robot]]) - np.matmul(self.utils.MatrixPower(self.F, self.assignment.STEPS), np.array([self.assignment.INIT_POSE[robot], self.assignment.INIT_VEL[robot]])))[0][k]
                    self.A[robot][self.assignment.DIMENSION*2 + k][k*self.iterations + time] = np.matmul( self.utils.MatrixPower(self.F, self.assignment.STEPS - 1 - time), self.G)[1][0]
                    self.B[robot][self.assignment.DIMENSION*2 + k] = (np.array([self.assignment.OptimalTargets[robot], self.assignment.TARGET_VEL[robot]]) - np.matmul(self.utils.MatrixPower(self.F, self.assignment.STEPS), np.array([self.assignment.INIT_POSE[robot], self.assignment.INIT_VEL[robot]])))[1][k]

    def inequality_contraints(self): #x_nominal : ndarray
        # sum(u[j] from 0 to T-1) < U_max * T
        # collision avoidance constraint; Compute distance between all robots (for nearby robots); for priority i<j;  
        # more [j] of robot -> more priority
        # bound on control input u; define U_max
        
        for robot in range(self.assignment.num_robots):
            for robot2 in range(0,robot):    
                for time in range(1, self.iterations-1):

                    for k in range(self.assignment.DIMENSION):
                        self.h[robot][((robot-1)*robot//2 + robot2) + time*(self.assignment.num_robots-1)*self.assignment.num_robots//2] = -self.params.R_COL*np.linalg.norm(self.x_nominal[time][:, robot][0]- self.x_nominal[time][:, robot2][0])  -  (np.array([self.x_nominal[time][:, robot][0]- self.x_nominal[time][:, robot2][0]]) @ np.array([self.x_nominal[time][:, robot2][0]]).T)[0][0]   +   (np.array([self.x_nominal[time][:, robot][0]- self.x_nominal[time][:, robot2][0]]) @ np.array([np.matmul( self.utils.MatrixPower(self.F, time), np.array([self.assignment.INIT_POSE[robot], self.assignment.INIT_VEL[robot]]))[0]]).T)[0][0]
                        for tmp in range(0, time):
                            self.g[robot][((robot-1)*robot//2 + robot2) + time*(self.assignment.num_robots-1)*self.assignment.num_robots//2][k*self.iterations + tmp] = -((self.x_nominal[time][:, robot] - self.x_nominal[time][:, robot2]).T @  np.row_stack((np.matmul( self.utils.MatrixPower(self.F, time - 1 - tmp), self.G)[0],np.matmul( self.utils.MatrixPower(self.F, time - 1 - tmp), self.G)[1])))[k][0]  #, np.matmul( MatrixPower(F, time - 1 - tmp), G)[1]))  #################DIMENSION 3

    def contraint_matrices(self):
        import cvxpy as cp
        self.obejective_function()
        self.equality_contraints()
        self.inequality_contraints()
        
        P = np.identity(np.shape(self.u_nominal)[0]*np.shape(self.u_nominal)[2]) 
        q = np.zeros(np.shape(self.u_nominal)[0]*np.shape(self.u_nominal)[2]) 
        
        prob = {}
        for i in range(self.assignment.num_robots):
            # print(np.shape(u[i]), np.shape(P))
            prob[i] = cp.Problem(cp.Minimize((1/2)*cp.quad_form(self.u_input[i], P) + q.T @ self.u_input[i]),
                [self.g[i] @ self.u_input[i] <= self.h[i],
                self.A[i] @ self.u_input[i] == self.B[i]]) #,
                #   u_in[i] <= 5,
                #   -5 <= u_in[i]])
            
            prob[i].solve()

    def get_optimal_traj_iteration_and_plot(self):
        # global self.assignment.DIMENSION
        x = self.assignment.INIT_POSE
        v = self.assignment.INIT_VEL
        Array = [(x,v)]
        X = [x]
        y_traj = [x[:,1]] ##REM TO CHANGE [:,1]
        x_traj = [x[:,0]] ##REM TO CHANGE [:,1]
        z_traj = [x[:,2]]  ############################DIM
        #u_nom = []
        
        for i in range(self.iterations-1):
            k=1
            u_fixed_t = np.zeros((self.assignment.num_robots, self.assignment.DIMENSION))
            u_robot = None
            u = None
            for robot in range(self.assignment.num_robots):
                for k in range(self.assignment.DIMENSION):
                    u_fixed_t[robot][k] = self.u_input[robot].value[k*self.iterations + i]
                u_robot = u_fixed_t[robot][0]

                for k in range(1, self.assignment.DIMENSION):
                    u_robot = np.column_stack((u_robot, u_fixed_t[robot][k]))
                if robot == 0:
                    u = u_robot
                else:
                    u = np.row_stack((u, u_robot))

            #u_nom.append(u)
            Array.append(self.utils.model(x, v, u))
            x = Array[-1][0]
            v = Array[-1][1]
            # print(Array[0])
            X.append(Array[-1][0])
            y_traj.append(X[-1][:,1]) ##REM TO CHANGE [:,1]
            x_traj.append(X[-1][:,0]) ##REM TO CHANGE [:,1]
            z_traj.append(X[-1][:,2]) ################################## DIM

        #u_nom.append(u*0)
        y_traj = np.array(y_traj)
        x_traj = np.array(x_traj)
        z_traj = np.array(z_traj)

        ########################################## DIM 3d
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(x_traj, y_traj, z_traj)
        # plt.show()

        # # return x_traj, y_traj, z_traj

        # self.assignment.utils.animation(x_traj, y_traj, z_traj)
        return np.array(Array)#, np.array(u_nom)


    def SCP_algorithm(self):
        error_SCP = []
        while True:        
            self.contraint_matrices()
            self.Trajectory = self.get_optimal_traj_iteration_and_plot() #x, u
            if (np.linalg.norm(self.Trajectory - self.x_nominal) < self.params.EPSILON_SCP):
                error_SCP.append(np.linalg.norm(self.Trajectory - self.x_nominal))
                break
            else:
                error_SCP.append(np.linalg.norm(self.Trajectory - self.x_nominal))
                self.x_nominal = np.copy(self.Trajectory)
        
        print(self.Trajectory[:, 0, :, 0]) # timestep, x|v, robot, dimension

        plt.plot(np.linspace(1, len(error_SCP),len(error_SCP)), error_SCP)
        plt.xlabel("Iterations")
        plt.ylabel("Error between current and previous nominal trajectory")
        plt.title("Error in SCP v/s Iterations")
        plt.plot(np.linspace(1, len(error_SCP),len(error_SCP)), np.zeros(len(error_SCP)))
        plt.show()

    def drone_control_and_final_plots(self):
        import matplotlib.cm as cm
        import mpl_toolkits.mplot3d.axes3d as p3

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = cm.rainbow(np.linspace(0, 1, self.params.number_robots*self.iterations))
        ax.scatter(self.Trajectory[:,0,:,0].T, self.Trajectory[:,0,:,1].T, self.Trajectory[:,0,:,2].T, color=colors ) #, s=self.iterations*self.params.number_robots)
        plt.show()

        self.assignment.utils.animation(self.Trajectory[:,0,:,0], self.Trajectory[:,0,:,1], self.Trajectory[:,0,:,2])

        def animate_scatters(iteration, data, scatters):
            """
            Update the data held by the scatter plot and therefore animates it.
            Args:
                iteration (int): Current iteration of the animation
                data (list): List of the data positions at each iteration.
                scatters (list): List of all the scatters (One per element)
            Returns:
                list: List of scatters (One per element) with new coordinates
            """
            for i in range(data[0].shape[0]):
                scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
            return scatters

        def main(data, save=False):
            """
            Creates the 3D figure and animates it with the input data.
            Args:
                data (list): List of the data positions at each iteration.
                save (bool): Whether to save the recording of the animation. (Default to False).
            """

            # Attaching 3D axis to the figure
            fig = plt.figure()
            ax = p3.Axes3D(fig)

            # Initialize scatters
            scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

            # Number of iterations
            iterations = len(data)

            # Setting the axes properties
            ax.set_xlim3d([self.params.AxesLimits[0][0], self.params.AxesLimits[0][1]])
            ax.set_xlabel('X')

            ax.set_ylim3d([self.params.AxesLimits[1][0], self.params.AxesLimits[1][1]])
            ax.set_ylabel('Y')

            ax.set_zlim3d([self.params.AxesLimits[2][0], self.params.AxesLimits[2][1]])
            ax.set_zlabel('Z')

            ax.set_title('3D Animated Scatter Example')

            # Provide starting angle for the view.
            ax.view_init(25, 10)

            ani = FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                            interval=50, blit=False, repeat=True)

            # if save:
            #     Writer = animation.writers['ffmpeg']
            #     writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
            #     ani.save('3d-scatted-animated.mp4', writer=writer)

            plt.show()


        data = self.Trajectory[:, 0, :, :]
        main(data, save=True)


if __name__=='__main__':
    params = PARAMS()
    utils = Utils(params)
    Assignment = assignment(params, utils)
    x_nom, u_nom = Assignment.get_nominal_trajectory_and_plots()
    Traj = OptimalTrajectory(x_nom, u_nom, Assignment)
    Traj.SCP_algorithm()
    Traj.drone_control_and_final_plots()