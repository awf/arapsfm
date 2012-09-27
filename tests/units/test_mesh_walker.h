#ifndef __TEST_MESH_WALKER_H__
#define __TEST_MESH_WALKER_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Geometry/mesh_walker.h"
#include "Util/pyarray_conversion.h"

// apply_displacement
void apply_displacement(PyArrayObject * npy_V,
                        PyArrayObject * npy_T,
                        PyArrayObject * npy_U,
                        PyArrayObject * npy_L,
                        PyArrayObject * npy_delta)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);
    PYARRAY_AS_MATRIX(double, npy_delta, delta_);

    VectorArrayAdapter<double> delta(delta_.num_rows(), delta_.num_cols(), delta_[0]);

    Mesh mesh(V.num_rows(), T);
    
    MeshWalker walker(mesh, V);
    walker.applyDisplacement(U, L, delta);
}


#endif
