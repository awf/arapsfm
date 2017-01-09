# transfer_user_constraints.py

# Imports
import os
import numpy as np
import argparse
from scipy.linalg import block_diag
from mesh.faces import faces_from_cell_array
from pprint import pprint

# right_multiply_affine_transform
def right_multiply_affine_transform(V0, V):
    t0 = np.mean(V0, axis=0)
    t = np.mean(V, axis=0)

    V0 = V0 - t0
    A = block_diag(V0, V0, V0)
    A_T = np.transpose(A)
    A_pinv = np.dot(np.linalg.inv(np.dot(A_T, A)), A_T)
    x = np.dot(A_pinv, np.ravel(np.transpose(V - t)))

    T = np.transpose(x.reshape(3, 3))
    d = t - np.dot(t0, T)

    return T, d

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh', type=str)
    parser.add_argument('input_constraints', type=str)
    parser.add_argument('output_mesh', type=str)
    parser.add_argument('output_constraints', type=str)

    args = parser.parse_args()
    pprint(args.__dict__)

    if os.path.isdir(args.input_constraints):
        input_constraint_files = os.listdir(args.input_constraints)

        input_constraints = map(
            lambda f: os.path.join(args.input_constraints, f),
            input_constraint_files)

        if not os.path.exists(args.output_constraints):
            os.makedirs(args.output_constraints)

        output_constraints = map(
            lambda f: os.path.join(args.output_constraints, f),
            input_constraint_files)
    else:
        input_constraints = [args.input_constraints]
        output_constraints = [args.output_constraints]

    z = np.load(args.input_mesh)
    print 'input_mesh:'
    print z.keys()
    V0 = z['points']
    z.close()

    # load output mesh `U0` and transform to `U`
    z = np.load(args.output_mesh)
    U0 = z['points']
    z.close()

    for input_, output in zip(input_constraints, output_constraints):
        # load input model points `V0`, and `V` from constraints
        print '<- %s' % input_
        z = np.load(input_)
        print 'input_constraints:'
        print z.keys()
        V = z['V']
        input_constraints = {k:z[k] for k in z.keys()}
        z.close()

        # calculate points `V0` -> `V`
        A, d = right_multiply_affine_transform(V0, V)
        print 'V0 -> V:'
        print np.around(A, decimals=3)
        print d

        print 'allclose? ', np.allclose(np.dot(V0, A) + d, V, atol=1e-3)
        U = np.dot(U0, A) + d

        # calculate Euclidean distance matrix from `V0` to `U0`
        D = V0[:, np.newaxis, :] - U0
        D = np.sum(D * D, axis=-1)
        
        # get argument distance transform
        arg_D = np.argmin(D, axis=1).astype(np.int32)

        # propagate constraints accordingly
        C1 = arg_D[input_constraints['C']]

        # output dictionary
        output_constraints=dict(
            C=C1,
            point_ids=C1,
            P=input_constraints['positions'],
            positions=input_constraints['positions'],
            T=input_constraints['T'],
            V=U)

        print '-> %s' % output
        np.savez_compressed(output, **output_constraints)

if __name__ == '__main__':
    main()

