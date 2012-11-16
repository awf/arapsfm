# check_mesh.py

# Imports
import argparse
import numpy as np
from vtk_ import iter_vtkCellArray

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    z = np.load(args.input)
    V = z['points']
    print '# Vertices:', V.shape[0]

    faces = list(iter_vtkCellArray(z['cells']))
    print '# Faces:', len(faces)

    n = np.unique(map(len, faces))
    print '# Vertices of on each face:', n

    edges = {}
    for f, face in enumerate(faces):
        for l in xrange(len(face)):
            i, j = face[l-1], face[l]
            edge = (i,j) if i < j else (j,i)
            edges.setdefault(edge, []).append(f)

    m = np.unique(map(len, edges.itervalues()))
    print '# Adjacent faces to each edge:', m

if __name__ == '__main__':
    main()

