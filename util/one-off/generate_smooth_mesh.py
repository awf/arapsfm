# generate_smooth_mesh.py

# Imports
from vtk_ import *

# main
def main():
    z = np.load('data/models/chihuahua.npz')
    V, T_ = z['points'], z['cells']

    poly_data = numpy_to_vtkPolyData(V, T_)

    # deci = vtk.vtkDecimatePro()
    # deci.SetInput(poly_data)
    # deci.SetTargetReduction(0.95)
    # deci.PreserveTopologyOn()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetNumberOfIterations(30)
    smooth.SetRelaxationFactor(0.2)
    smooth.SetInput(poly_data)
    smooth.SetInput(poly_data)
    smooth.Update()
    print smooth
    view_vtkPolyData(smooth.GetOutput())

    output_poly_data = smooth.GetOutput()
    V, T_ = vtkPolyData_to_numpy(output_poly_data)
    np.savez_compressed('data/models/chihuahua_smoothed.npz',
                        points=V,
                        cells=T_)
    return

    sinc = vtk.vtkWindowedSincPolyDataFilter()
    sinc.SetNumberOfIterations(50)
    sinc.SetInput(poly_data)
    sinc.Update()
    print sinc
    view_vtkPolyData(sinc.GetOutput())

if __name__ == '__main__':
    main()

