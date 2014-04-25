import sys
import numpy as np
import matplotlib.pyplot as plt

F2D_3 = [0, 1, 2]
F2D_4 = [0, 1, 2, 3]
F2D_6 = [0, 3, 1, 4, 2, 5]
F2D_8 = [0, 4, 1, 5, 2, 6, 3, 7]
F3D_4 = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
F3D_8 = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
         [1, 2, 6, 5], [2, 6, 7, 3], [3, 7, 4, 0]]
F3D_10 = [[0, 4, 1, 5, 2, 6], [0, 7, 3, 8, 1, 4],
          [1, 8, 3, 9, 2, 5], [2, 9, 3, 7, 0, 6]]
F3D_20 = [[0, 8, 1, 9, 2, 10, 3, 11], [4, 15, 7, 14, 6, 13, 5, 12],
          [0, 16, 4, 12, 5, 17, 1, 8], [1, 17, 5, 13, 6, 18, 2, 9],
          [2, 18, 6, 14, 7, 19, 3, 10], [3, 19, 7, 15, 4, 16, 0, 11]]


def plot_mesh_from_ui(ui, stats=True, color="g"):
    """Plot the mesh from the UserInput instance

    """

    ncoord = ui.mesh.ncoord()
    coords = ui.mesh.nodes()
    nnode = ui.mesh.nnodes()
    connect = ui.mesh.connect()
    nelnodes = ui.mesh.nelnodes()
    elements = ui.mesh.elements()
    nels = ui.mesh.nels()

    sidesets = ui.mesh.sideset("all")
    nodesets = ui.mesh.nodeset("all")
    if stats:
        # print out mesh stats
        print "SIDESET SUMMARY"
        for sid, sideset in sidesets.items():
            print "  SIDESET {0}".format(sid)
            print "    {0}".format("\n    ".join(
                    "ELEMENT: {0}, FACE: {1}".format(x, y) for (x, y) in sideset))

        print "\n\nNODESET SUMMARY"
        for (nid, nodes) in nodesets.items():
            print "  NODESET {0}".format(nid)
            print "    NODES: {0}\n".format(", ".join(repr(x) for x in nodes))

    plot_mesh(coords, ncoord, nnode, connect, nels, elements, nelnodes, color,
              nodesets=nodesets, sidesets=sidesets)

def plot_mesh(coords, ncoord, nnode, connect, nels, elements, nelnodes, color,
              nodesets=None, sidesets=None):
    """Function to plot a mesh

    """
    if ncoord == 2:
        # Plot a 2D mesh
        for lmn in range(nels):
            x = np.zeros((max(nelnodes), 2))
            labels = []
            for i in range(nelnodes[lmn]):
                node = connect[lmn, i]
                labels.append("{0:0{1}d}".format(node, len(str(nnode))))
                x[i, 0:2] = coords[node, 0:2]
                continue
            plt.scatter(x[:, 0], x[:, 1], c="r")

            # label nodes
            for label, xx, yy in zip(labels, x[:, 0], x[:, 1]):
                plt.annotate(label, xy=(xx, yy), xytext=(-5, 5),
                             textcoords="offset points", ha="right", va="bottom")

            # label element at center
            x0, x1 = x[0, 0], x[1, 0]
            y0, y1 = x[1, 1], x[2, 1]
            xc = x0 + (x1 - x0) / 2.
            yc = y0 + (y1 - y0) / 2.
            plt.annotate("{0}".format(lmn), xy=(xc, yc))

        if nodesets is not None:
            # plot the nodesets
            for (nid, nodeset) in nodesets.items():
                x = coords[nodeset]
                plt.scatter(x[:, 0], x[:, 1], label="Nodeset {0}".format(nid))

        if sidesets is not None:
            for sid, sideset in sidesets.items():
                c = 'r'
                for i, (el, face) in enumerate(sideset):
                    label = "Sideset {0}".format(sid) if i == 0 else None
                    nodes = {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 0]}[face]
                    conn = connect[el]
                    x = coords[conn[nodes]]
                    plt.plot(x[:, 0], x[:, 1], c=c, label=label)

        if sidesets is not None or nodesets is not None:
            plt.legend(loc="best")
        plt.show()

    elif ncoord == 3:
        # Plot a 3D mesh
        for lmn in range(nels):
            for i in range(nelnodes[lmn]):
                x[i, 0:3] = coords[0:3, connect[i, lmn]]
                continue
            scatter3(x[:,0], x[:, 1], x[:, 2], 'MarkerFaceColor', 'r')

            if nelnodes[lmn] == 4:
                patch('Vertices', x, 'Faces', F3D_4, 'FaceColor', 'none',
                      'EdgeColor', color);
            elif nelnodes[lmn] == 10:
                patch('Vertices', x, 'Faces', F3D_10, 'FaceColor', 'none',
                      'EdgeColor', color);
            elif nelnodes[lmn] == 8:
                patch('Vertices', x, 'Faces', F3D_8, 'FaceColor', 'none',
                      'EdgeColor', color);
            elif nelnodes[lmn] == 20:
                patch('Vertices', x, 'Faces', F3D_20, 'FaceColor', 'none',
                      'EdgeColor', color);
            continue

#    axis equal
#    hold off

    return
