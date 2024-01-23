"""
This file contains the Truss class, which is used to calculate the statics of a truss structure either linearly or
nonlinearly.
"""

# Import standard library modules

# Import modules
import lattice
import numpy as np
import scipy.sparse as ss  # sparse matrix


class Truss(lattice.Lattice):
    """
    Truss class
    ===========
        This class is used to calculate the statics of a truss structure linearly
    """
    calForces, calDisps = None, None
    extForces, extDisps = None, None
    # K: stiffness matrix, A: structure matrix,
    # Aidx: structure matrix index, Aval: structure matrix value
    K, A, Aidx, Aval = None, None, [], []

    def __init__(self, node_num_4_each_edge: int = 3, path=None):
        """
        Truss constructor
        ===================
        :param node_num_4_each_edge: int
            Number of nodes for each edge of the lattice.
        """
        super().__init__(node_num_4_each_edge, path)  # call the constructor of the parent class
        self.extForces = np.zeros((self.nodes.shape[0], 2))
        self.extDisps = np.zeros((self.nodes.shape[0], 2)) * np.nan  # nan means free
        #
        self.calc_structure_sparse_matrix()
        self.calc_sparse_stiffness_matrix()

    def calc_structure_sparse_matrix(self):
        """
        Calculate the global structure matrix of the truss. The structure matrix is a special form of the stiffness
        matrix (with bond stiffness set to 1).
        ==================
        :return:
        """
        # Add the structure matrix index of each element to the global structure matrix index
        # if not self.Aidx:
        self.Aidx = []
        for i, j in self.elements:
            self.Aidx += [[i * 2, i * 2], [i * 2, i * 2 + 1], [i * 2 + 1, i * 2], [i * 2 + 1, i * 2 + 1],
                          [i * 2, j * 2], [i * 2, j * 2 + 1], [i * 2 + 1, j * 2], [i * 2 + 1, j * 2 + 1],
                          [j * 2, i * 2], [j * 2, i * 2 + 1], [j * 2 + 1, i * 2], [j * 2 + 1, i * 2 + 1],
                          [j * 2, j * 2], [j * 2, j * 2 + 1], [j * 2 + 1, j * 2], [j * 2 + 1, j * 2 + 1]]
        # Add the structure matrix value of each element to the global structure matrix value
        self.Aval = []
        for i, j in self.elements:
            dist = np.linalg.norm(self.nodes[i, :] - self.nodes[j, :])
            c = (self.nodes[j, 0] - self.nodes[i, 0]) / dist
            s = (self.nodes[j, 1] - self.nodes[i, 1]) / dist
            self.Aval += [c * c, c * s, s * c, s * s, -c * c, -c * s, -s * c, -s * s,
                          -c * c, -c * s, -s * c, -s * s, c * c, c * s, s * c, s * s]

        # Create the global structure matrix with coo_matrix
        # slice the first column of Aidx
        __row = np.array(self.Aidx)[:, 0]
        __col = np.array(self.Aidx)[:, 1]

        self.A = ss.coo_matrix((self.Aval, (__row, __col)),
                               shape=(self.nodes.shape[0] * 2, self.nodes.shape[0] * 2)).tocsr()

    def add_extForces(self, nodes: np.ndarray = None, forces: np.ndarray = None, reset: bool = False):
        """
        Add external forces to the truss.
        Zero forces are added to the nodes that are not specified.
        ==================
        :param nodes: shape = (n, )
        :param forces: shape = (2, ) or (n, 2)
        :param reset: reset external forces to zero
        :return:
        """
        if reset:
            self.extForces *= 0
        if nodes is not None:
            self.extForces[nodes] = forces

    def add_extDisps(self, nodes: np.ndarray = None, disps: np.ndarray = None, reset: bool = False):
        """
        Add external displacements to the truss.
        NaN displacements are added to the nodes that are not specified.
        NaN displacements are free nodes.
        ==================
        :param nodes:
        :param disps:
        :param reset: reset external displacements to NaN
        :return:
        """
        if reset:
            self.extDisps *= np.nan
        if nodes is not None:
            self.extDisps[nodes] = disps

    def calc_sparse_stiffness_matrix(self):
        """
        Calculate the global stiffness matrix of the truss.
        The stiffness matrix is a special form of the structure.
        ==================
        :return:
        """
        __Kval = self.Aval * np.repeat(self.weights, 16)
        __row = np.array(self.Aidx)[:, 0]
        __col = np.array(self.Aidx)[:, 1]
        self.K = ss.coo_matrix((__Kval, (__row, __col)),
                               shape=(self.nodes.shape[0] * 2, self.nodes.shape[0] * 2)).tocsr()

    def linearSolve(self):
        """
        Solve the linear system of equations.
        ==================
        :return: bond stretch, bond inner force

        """
        self.calDisps = self.extDisps.copy().flatten()
        self.calForces = self.extForces.copy().flatten()
        # Find free and fixed nodes using boolean indexing
        free_nodes = np.isnan(self.extDisps.flatten())
        # print(free_nodes)
        fixed_nodes = ~free_nodes

        # Extract submatrices of K and F corresponding to free nodes
        K_free = self.K[free_nodes][:, free_nodes]
        F_free = (self.extForces.flatten() - self.K[:, fixed_nodes] @ (self.extDisps.flatten()[fixed_nodes]))[
            free_nodes]

        # Solve for displacements of free nodes
        U_free = ss.linalg.spsolve(K_free, F_free)

        # Assign calculated displacements to calDisps
        self.calDisps[free_nodes] = U_free

        # Calculate forces
        self.calForces = self.K.dot(self.calDisps)

        # Reshape calDisps and calForces
        self.calDisps = self.calDisps.reshape((-1, 2))
        self.calForces = self.calForces.reshape((-1, 2))

        #
        return self.calc_bond_stretch()

    def calc_bond_stretch(self):
        """
        Calculate the stress of each element.
        :return: bond stretch, bond inner force
        """
        __idx = self.elements[:, 0]
        __idy = self.elements[:, 1]
        __disps_diff = self.calDisps[__idy, :] - self.calDisps[__idx, :]  # displacement difference
        #
        __bond_vector = self.nodes[__idy, :] - self.nodes[__idx, :]  # bond vector
        __bond_length = np.linalg.norm(__bond_vector, axis=1)  # bond length
        __bond_direction = __bond_vector / __bond_length[:, np.newaxis]  # bond direction
        #
        __bond_stretch = np.sum(__disps_diff * __bond_direction, axis=1)  # bond stretch
        #
        __bond_inner_force = __bond_stretch * self.weights  # bond inner force
        #
        return __bond_stretch, __bond_inner_force


def test_truss():
    """
    Test the Truss class
    ===================
    :return:
    """
    # Create a truss object
    truss = Truss(node_num_4_each_edge=2, path="./TRUSS_TEST/")
    # Add external constraints
    truss.add_extForces(np.array([-1]), np.array([[0.1, 0]]), reset=True)
    truss.add_extDisps(np.array([0, 1]), np.array([[0, 0], [np.nan, 0]]), reset=True)
    stretch, force = truss.linearSolve()
    # Print results
    print("Bond Location:\n", truss.elements)
    print("Bond Length Change:\n", stretch)
    print("Bond Force:\n", force)
    print("Node Force:\n", truss.calForces)
    print("Node Displacement:\n", truss.calDisps)
    # Plot results
    truss.plot_lattice()


if __name__ == "__main__":
    test_truss()
