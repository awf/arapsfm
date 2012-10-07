# visualise_mesh.py

# Imports
import numpy as np
import subprocess
from mesh import faces

# main
def main():
    output = 'working/model_to_visualise.npz'
    z = np.load('data/models/Cheetah_4.npz')
    V = z['points']
    T_ = z['cells']

    T = faces.faces_from_cell_array(T_)

    np.savez_compressed(output, V=V, T=T)

    args = ['python', 'visualise/visualise_standalone.py', output,
            '-c', 'SetParallelProjection=True']
    print ' '.join(args)

    subprocess.check_call(args)

if __name__ == '__main__':
    main()

