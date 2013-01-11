# solve_core_recovery.py

# Imports
from util.cmdline import *
from visualise.visualise import *

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('solver')
    args = parser.parse_args()

    solver = load(args.solver)

    for i in xrange(2):
        callback, states = solver.solve_instance_callback(i)

        solver.solve_instance(i, no_silhouette=True,
                              fixed_scale=True,
                              callback=callback)

        solver.solve_silhouette(i)

        solver.solve_instance(i, no_silhouette=False,
                              fixed_scale=False,
                              callback=callback)

        # pickle_.dump('states.dat', states)

        if i + 1 < solver.n:
            solver._s.V1[i + 1] = solver._s.V1[i]

    callback, states = solver.solve_core_callback()
    solver.solve_core(callback=callback)
    pickle_.dump('states.dat', states)

    return

    j = 0
    vis = VisualiseMesh()
    vis.add_mesh(solver._s.V1[j], solver.T)
    vis.add_image(solver.frames[j])
    vis.add_projection(solver.C[j],
                       solver.P[j])

    vis.add_quick_silhouette(solver.silhouette_preimages(j), solver.S[j])
    vis.camera_actions(('SetParallelProjection', (True,)))
    vis.execute()

if __name__ == '__main__':
    main()

