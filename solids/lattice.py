"""
Lattice.py
==========
This module contains the Lattice class, which is used to generate a lattice
structure.
"""

# Import standard library modules
import pygmsh
import numpy as np
import itertools
import matplotlib.pyplot as plt
import os


class Lattice:
    """
    Lattice class
    =============
    This class is used to generate a lattice structure.
    :param nodes : numpy array
        List of nodes in the lattice.
            Nodes[i] = [x, y, z]
    :param elements : numpy array
        List of elements(edges) in the lattice.
            Elements[i] = [node1, node2]
    :param weights : numpy array
        List of element weights in the lattice.
            Weights[i] = Kij
    :param msh : Mesh object
        Mesh object used to generate the lattice.
    :param path : str
        Path to save the lattice. If None, do not save.
    """
    nodes, elements, weights = None, None, None
    msh = None
    path = ''

    def __init__(self, node_num_4_each_edge: int = 3, path=None):
        """
        Lattice constructor
        ===================
        This method initializes the Lattice object.
        :param node_num_4_each_edge: int
            Number of nodes for each edge of the lattice.
        """
        if path is not None:
            self.path = path
            os.makedirs(self.path, exist_ok=True)

        # msh is a Mesh object
        with pygmsh.geo.Geometry() as geom:
            # http://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
            geom.add_rectangle(
                -.5, +.5, -.5, +.5, 0, mesh_size=1 / (node_num_4_each_edge - 1), make_surface=True,
            )
            self.msh = geom.generate_mesh(dim=2)
        # Nodes is a numpy array
        self.nodes = np.array(self.msh.points)[:, :2]
        # Elements is a numpy array
        cells = self.msh.cells_dict['triangle'].tolist()
        ##
        tmp = []
        for cell in cells:
            tmp.extend(list(itertools.combinations(cell, 2)))
        ##
        dupl_edges = []
        for (m, n) in tmp:
            dupl_edges.extend([(m, n), (n, m)])
        dupl_edges = list(set(dupl_edges))  # remove duplicates
        ##
        single_edges = []
        for (m, n) in dupl_edges:
            if m < n:
                single_edges.extend([(m, n)])
        self.elements = np.array(single_edges)
        # Weights is a numpy array
        self.weights = np.ones((len(self.elements),))

    def plot_lattice(self):
        """
        Plot lattice
        ============
        This method plots the lattice.
        """
        plt.figure(figsize=(8 / 2.54, 8 / 2.54))
        plt.title("Lattice", fontsize=18)
        plt.gca().set_aspect("equal")
        plt.xlim(-0.7, 0.7)
        plt.ylim(-0.7, 0.7)
        for (i, j), w in zip(self.elements, self.weights):
            plt.plot(self.nodes[[i, j], 0], self.nodes[[i, j], 1], color="green", linewidth=2*w)
        plt.scatter(self.nodes[:, 0], self.nodes[:, 1], color="blue", s=50)
        if self.path is not None:
            plt.savefig(self.path + "lattice.svg", dpi=300)
        plt.show()

    def save_lattice(self):
        """
        Save the lattice.
        :return:
        """
        print("Saving lattice...")

        np.savez(self.path + "lattice.npz", nodes=self.nodes,
                 elements=self.elements,
                 weights=self.weights)

    def load_lattice(self, pth: str = None):
        """
        Load the lattice.
        :param pth: str
        :return:
        """
        if pth is not None:
            self.path = pth
        data = np.load(self.path + "lattice.npz")
        print(data)
        self.nodes = data["nodes"]
        self.elements = data["elements"]
        self.weights = data["weights"]


def lattice_test():
    print("Lattice test")
    lat = Lattice(node_num_4_each_edge=10, path="./LATTICE_TEST/")
    lat.plot_lattice()
    lat.save_lattice()
    lat.load_lattice()


if __name__ == "__main__":
    # Execute only if run as a script
    # Otherwise, do not execute
    # Print out a lattice figure
    lattice_test()
