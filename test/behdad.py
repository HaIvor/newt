import random
import time
import numpy as np

class Node:
    """Representing the node and its actions and attributes. This class inherits the Thread class, so each instance
    of this class would be executed in a separate thread. This is important in order of improving the simulation
    speed and also simulating asynchronous communications """

    def __init__(self, node_id: int, x0: np.array, epsilon: float, c: float, costfun: str,
                 minimum_accepted_divergence: float,
                 adjacency_vector: np.array,
                 simulation_function_xtx_btx):

        super(Node, self).__init__()

        self.node_id = node_id
        self.epsilon = epsilon
        self.xi = np.array(x0) + np.random.uniform(-1, 1, x0.size)

        # generate random cost function
        cost_functions = {
            'quad': self.quadratic(),
            'exp' : self.exponential()
        }
        self.A, self.b = cost_functions[costfun]
        self.identifier = costfun

        # list for storing evolution of signals
        self.all_calculated_xis = []
        self.evolution_costfun = []
        self.ratio_evol = []
        self.zi_evol = []



        self.simulation_function_xtx_btx = simulation_function_xtx_btx

        self.ff = simulation_function_xtx_btx.get_fn(self.xi, self.A, self.b, self.identifier)

        self.number_of_neighbors = np.sum(adjacency_vector)
        self.minimum_accepted_divergence = minimum_accepted_divergence
        self.adjacency_vector = adjacency_vector

        self.hi_old = np.eye(x0.size)
        self.hi = np.eye(x0.size)

        self.gi = np.zeros(x0.size).transpose()
        self.gi_old = np.zeros(x0.size).transpose()
        self.zi = np.eye(x0.size) #self.hi_old
        self.yi = np.zeros(x0.size).transpose() #self.gi_old

        self.c = c
        self.cI = c * np.eye(x0.size)  # variable to be check in order to ensure robustness and large basin of attraction


        self.sigma_yi = np.zeros(x0.size).transpose()  # counter of the total mass-y sent
        self.sigma_zi = np.zeros((x0.size, x0.size))    # counter of the total mass-z sent, matrix x0.size X x0.size

        self.rho_yj = np.zeros((adjacency_vector.size, x0.size))   # counter of the total mass-y received from j, matrix_dim X xo.size
        self.rho_yj_old = np.zeros((adjacency_vector.size, x0.size))
        self.rho_zj = np.zeros((adjacency_vector.size, x0.size, x0.size))  # counter of the total mass-z received from j
        self.rho_zj_old = np.zeros((adjacency_vector.size, x0.size, x0.size))

        self.iter = 0

        self.msg_rel = True  # flag to simulate a packet loss

        self.is_ready_to_receive = False
        self.is_ready_to_update = False
        self.is_ready_to_transmit = True

        self.ratio = 0

    def quadratic(self):
        '''while True:
            A = np.random.uniform(0, 1, (self.xi.size, self.xi.size))
            A = 0.5 * (A + A.transpose())  # make the matrix symmetric
            if np.linalg.det(A) >= 1:   # ensure positive definiteness
                break'''
        A = np.random.uniform(0, 0.3, self.xi.size) * np.eye(self.xi.size)
        b = np.random.uniform(0, 2, self.xi.size).transpose()
        return A, b

    def exponential(self):
        A = np.zeros((2, self.xi.size, self.xi.size))
        b = np.zeros((2, self.xi.size))
        for i in range(2):
            '''while True:
                AA = np.random.uniform(-2, 3, (self.xi.size, self.xi.size))
                AA = 0.5 * (AA + AA.transpose())  # make the matrix symmetric
                if np.linalg.det(AA) >= 1:  # ensure positive definiteness
                    break'''
            AA = np.random.uniform(0, 0.3, self.xi.size) * np.eye(self.xi.size)
            bb = np.random.uniform(0, 1, self.xi.size)
            A[i] = AA
            b[i] = bb
        return A, b

    def transmit_data(self):
        """This method update the yi and zi in each iteration and create a message including the new updated yi and
        zi. Finally the new message will be broadcast to all neighbors of this node. """
        print(f"Node ID: {self.node_id} -  transmitting data started!\n")

        self.msg_rel = True  # always reset the simulation flag, starting point is reliable

        self.yi = (1 / (self.number_of_neighbors + 1)) * self.yi
        self.zi = self.zi / (self.number_of_neighbors + 1)

        self.sigma_yi = self.sigma_yi + self.yi
        self.sigma_zi = self.sigma_zi + self.zi

        # simulate packet loss (not considered for the moment)
        if random.randrange(0, 9, 1) == 1:  # 1/10 chance to loss message
            self.msg_rel = False
            print("message LOST")

        '''
        if random.randrange(0, 9, 1)%2 == 0:  # 1/2 chance to loss message
            self.msg_rel = False
            print("message LOST")
        '''
        print(f"Node ID: {self.node_id} -  transmitting data ended!\n")
        return

    '''def broadcast(self, message):
        """This method will broadcast the passed message to all neighbors of this node."""
        print(f"Node ID: {self.node_id} -  broadcasting started!\n")
        with self.lock:
            i = 0
            while i < len(self.all_nodes_message_buffers):
                if self.adjacency_vector[i] == 1:
                    self.all_nodes_message_buffers[i].put(message)
                i += 1
        time.sleep(0.05)
        print(f"Node ID: {self.node_id} -  broadcasting ended!\n")
        return'''

    def receive_data(self, message):
        """This method will handle the reception of data from neighbors. Using the yi and zi from the messages,
        the yi and zi would be updated by this method."""
        print(f"Node ID: {self.node_id} -  receiving data started!\n")
        self.j = message['node_id']

        # update old virtual mass received
        self.rho_zj_old[self.j] = self.rho_zj[self.j]
        self.rho_yj_old[self.j] = self.rho_yj[self.j]

        if message['msg_rel']:       # update virtual mass of neighbour
            self.rho_yj[self.j] = message['sigma_yi']
            self.rho_zj[self.j] = message['sigma_zi']

        # update estimate
        self.yi = self.yi + self.rho_yj[self.j] - self.rho_yj_old[self.j]
        self.zi = self.zi + self.rho_zj[self.j] - self.rho_zj_old[self.j]

        print(f"Node ID: {self.node_id} -  Receiving data ended!\n")
        return

    def update_estimation(self, iter):
        """This method will calculate the next xi and also update the hi and gi by using the new xi. """
        print(f"Node ID: {self.node_id} -  Updating data started!\n")
        self.all_calculated_xis.append(self.xi)
        self.evolution_costfun.append(self.ff)
        while iter > len(self.evolution_costfun):
            self.all_calculated_xis.append(self.xi)
            self.evolution_costfun.append(self.ff)


        if (np.abs(np.linalg.eigvals(self.zi)) < self.c).all():
            self.zi = self.cI

        self.xi = (1 - self.epsilon) * self.xi + np.matmul((self.epsilon * np.linalg.inv(self.zi)),
                                                               np.transpose(self.yi))

        self.ff = self.simulation_function_xtx_btx.get_fn(self.xi, self.A, self.b, self.identifier)

        self.gi_old = self.gi
        self.hi_old = self.hi

        self.hi = self.simulation_function_xtx_btx.get_hessian_fn(self.xi, self.A, self.b, self.identifier)
        self.gi = np.subtract(np.matmul(self.hi, self.xi.transpose()),
                              self.simulation_function_xtx_btx.get_gradient_fn(self.xi, self.A, self.b, self.identifier))

        self.yi = self.yi + self.gi - self.gi_old
        self.zi = self.zi + self.hi - self.hi_old


        print(f"Node ID: {self.node_id} -  Updating data ended!\n")

        return

    def has_result_founded(self):
        """This method will check and verify if the calculated xi in this node has sufficiently converged. If for the
        last calculated xi, the difference between xi and x(i-1) is less than minimum accepted divergence that has
        been provided by the user, then it would be considered that calculated xi has enough convergence to its
        target value. """
        self.is_convergence_sufficient = False
        if len(self.all_calculated_xis) > 400:
            self.is_convergence_sufficient = True
            for i in range(400):
                for j in range(self.all_calculated_xis[-(i + 1)].size):
                    if abs(abs(self.all_calculated_xis[-(i + 1)][j]) - abs(
                            self.all_calculated_xis[-(i + 1) - 1][j])) > self.minimum_accepted_divergence:
                        self.is_convergence_sufficient = False
        return self.is_convergence_sufficient

