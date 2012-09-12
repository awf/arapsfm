#ifndef __LM_SOLVERS__
#define __LM_SOLVERS__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Geometry/mesh.h"
#include "Energy/arap.h"
#include "Energy/projection.h"
#include "Energy/narrow_band_silhouette.h"
#include "Energy/laplacian.h"
#include "Solve/node.h"
#include "Solve/problem.h"
#include "Solve/optimiser_options.h"

#include "Util/pyarray_conversion.h"

// solve_single_arap_proj
int solve_single_arap_proj(PyArrayObject * npy_V,
                 PyArrayObject * npy_T,
                 PyArrayObject * npy_X,
                 PyArrayObject * npy_V1,
                 PyArrayObject * npy_C,
                 PyArrayObject * npy_P,
                 PyArrayObject * npy_lambdas,
                 const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_X, X);
    PYARRAY_AS_MATRIX(double, npy_V1, V1);
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);
    VertexNode * nodeV1 = new VertexNode(V1);
    RotationNode * nodeX = new RotationNode(X);

    Problem problem;
    problem.AddFixedNode(nodeV);
    problem.AddNode(nodeV1);
    problem.AddNode(nodeX);

    ARAPEnergy * arapEnergy = new ARAPEnergy(*nodeV, *nodeX, *nodeV1, mesh, sqrt(lambdas[0]));
    ProjectionEnergy * projEnergy = new ProjectionEnergy(*nodeV1, C, P, sqrt(lambdas[1]));

    problem.AddEnergy(arapEnergy);
    problem.AddEnergy(projEnergy);

    return problem.Minimise(*options);
}

// solve_single_lap_proj
int solve_single_lap_proj(PyArrayObject * npy_V,
                  PyArrayObject * npy_T,
                  PyArrayObject * npy_C,
                  PyArrayObject * npy_P,
                  PyArrayObject * npy_lambdas,
                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_VECTOR(int, npy_C, C);
    PYARRAY_AS_MATRIX(double, npy_P, P);
    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);

    Problem problem;
    problem.AddNode(nodeV);

    LaplacianEnergy * lapEnergy = new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[0]));
    ProjectionEnergy * projEnergy = new ProjectionEnergy(*nodeV, C, P, sqrt(lambdas[1]));

    problem.AddEnergy(lapEnergy);
    problem.AddEnergy(projEnergy);

    return problem.Minimise(*options);
}

// solve_single_lap_silhouette
int solve_single_lap_silhouette(PyArrayObject * npy_V,
                  PyArrayObject * npy_T,
                  PyArrayObject * npy_U,
                  PyArrayObject * npy_L,
                  PyArrayObject * npy_S,
                  PyArrayObject * npy_SN,
                  PyArrayObject * npy_lambdas,
                  PyArrayObject * npy_preconditioners,
                  int narrowBand,
                  const OptimiserOptions * options)
{
    PYARRAY_AS_MATRIX(double, npy_V, V);
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_U, U);
    PYARRAY_AS_VECTOR(int, npy_L, L);

    PYARRAY_AS_MATRIX(double, npy_S, S);
    PYARRAY_AS_MATRIX(double, npy_SN, SN);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    VertexNode * nodeV = new VertexNode(V);
    nodeV->SetPreconditioner(preconditioners[0]);

    MeshWalker meshWalker(mesh, V);
    BarycentricNode * nodeU = new BarycentricNode(U, L, meshWalker);
    nodeU->SetPreconditioner(preconditioners[1]);

    Problem problem;
    problem.AddNode(nodeV);
    problem.AddNode(nodeU);

    LaplacianEnergy * lapEnergy = new LaplacianEnergy(*nodeV, mesh, sqrt(lambdas[0]));

    SilhouetteProjectionEnergy * silProjEnergy = new SilhouetteProjectionEnergy(*nodeV, *nodeU, S, mesh,
        sqrt(lambdas[1]), narrowBand);

    SilhouetteNormalEnergy * silNormalEnergy = new SilhouetteNormalEnergy(*nodeV, *nodeU, SN, mesh,
        sqrt(lambdas[2]), narrowBand);

    problem.AddEnergy(lapEnergy);
    problem.AddEnergy(silProjEnergy);
    problem.AddEnergy(silNormalEnergy);

    return problem.Minimise(*options);
}

// solve_multiview_arap_silhouette
int solve_multiview_arap_silhouette(
    PyArrayObject * npy_T,
    PyArrayObject * npy_V,
    PyObject * list_multiX,
    PyObject * list_multiV,
    PyObject * list_multiU,
    PyObject * list_multiL,
    PyObject * list_multiS,
    PyObject * list_multiSN,
    PyArrayObject * npy_lambdas,
    PyArrayObject * npy_preconditioners,
    int narrowBand,
    const OptimiserOptions * options)
{
    // 
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
                            
    auto multiX = PyList_to_vector_of_Matrix<double>(list_multiX);
    auto multiV = PyList_to_vector_of_Matrix<double>(list_multiV);
    auto multiU = PyList_to_vector_of_Matrix<double>(list_multiU);
    auto multiL = PyList_to_vector_of_Vector<int>(list_multiL);
    auto multiS = PyList_to_vector_of_Matrix<double>(list_multiS);
    auto multiSN = PyList_to_vector_of_Matrix<double>(list_multiSN);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    Problem problem;

    // core vertices
    VertexNode * coreVertexNode = new VertexNode(V);
    problem.AddFixedNode(coreVertexNode);

    // instance vertices
    vector<VertexNode *> instVertexNodes;
    for (auto i = multiV.begin(); i != multiV.end(); i++)
    {
        instVertexNodes.push_back(new VertexNode(*(*i)));
        problem.AddNode(instVertexNodes.back());
    }
    instVertexNodes.back()->SetPreconditioner(preconditioners[0]);

    // instances rotations
    vector<RotationNode *> instRotationNodes;
    for (auto i = multiX.begin(); i != multiX.end(); i++)
    {
        instRotationNodes.push_back(new RotationNode(*(*i)));
        problem.AddNode(instRotationNodes.back());
    }
    instRotationNodes.back()->SetPreconditioner(preconditioners[1]);

    // instance barycentric coordinates
    vector<BarycentricNode *> instBarycentricNodes;
    vector<MeshWalker *> meshWalkers;

    for (int i = 0; i < multiU.size(); i++)
    {
        MeshWalker * meshWalker = new MeshWalker(mesh, *multiV[i]);
        meshWalkers.push_back(meshWalker);

        instBarycentricNodes.push_back(new BarycentricNode(*multiU[i], *multiL[i], *meshWalker));
        problem.AddNode(instBarycentricNodes.back());
    }
    instBarycentricNodes.back()->SetPreconditioner(preconditioners[2]);

    // add energies

    // arap
    for (int i = 0; i < instVertexNodes.size(); i++)
    {
        problem.AddEnergy(new ARAPEnergy(*coreVertexNode, *instRotationNodes[i], *instVertexNodes[i], 
                          mesh, sqrt(lambdas[0])));
    }

    // silhouette
    for (int i = 0; i < instVertexNodes.size(); i++)
    {
        problem.AddEnergy(new SilhouetteProjectionEnergy(*instVertexNodes[i], *instBarycentricNodes[i],
            *multiS[i], mesh, sqrt(lambdas[1]), narrowBand));

        problem.AddEnergy(new SilhouetteNormalEnergy(*instVertexNodes[i], *instBarycentricNodes[i],
            *multiSN[i], mesh, sqrt(lambdas[2]), narrowBand));
    }

    // minimise
    int ret = problem.Minimise(*options);

    // dealloc 
    dealloc_vector(meshWalkers);
    dealloc_vector(multiX);
    dealloc_vector(multiV);
    dealloc_vector(multiU);
    dealloc_vector(multiL);
    dealloc_vector(multiS);
    dealloc_vector(multiSN);

    return ret;
}

// solve_multiview_lap_silhouette
int solve_multiview_lap_silhouette(
    PyArrayObject * npy_T,
    PyArrayObject * npy_V,
    PyObject * list_multiX,
    PyObject * list_multiV,
    PyObject * list_multiU,
    PyObject * list_multiL,
    PyObject * list_multiS,
    PyObject * list_multiSN,
    PyArrayObject * npy_lambdas,
    PyArrayObject * npy_preconditioners,
    int narrowBand,
    bool uniformWeights,
    const OptimiserOptions * options)
{
    // 
    PYARRAY_AS_MATRIX(int, npy_T, T);
    PYARRAY_AS_MATRIX(double, npy_V, V);
                            
    auto multiX = PyList_to_vector_of_Matrix<double>(list_multiX);
    auto multiV = PyList_to_vector_of_Matrix<double>(list_multiV);
    auto multiU = PyList_to_vector_of_Matrix<double>(list_multiU);
    auto multiL = PyList_to_vector_of_Vector<int>(list_multiL);
    auto multiS = PyList_to_vector_of_Matrix<double>(list_multiS);
    auto multiSN = PyList_to_vector_of_Matrix<double>(list_multiSN);

    PYARRAY_AS_VECTOR(double, npy_lambdas, lambdas);
    PYARRAY_AS_VECTOR(double, npy_preconditioners, preconditioners);

    Mesh mesh(V.num_rows(), T);

    Problem problem;

    // core vertices
    VertexNode * coreVertexNode = new VertexNode(V);
    problem.AddNode(coreVertexNode);

    // instance vertices
    vector<VertexNode *> instVertexNodes;
    for (auto i = multiV.begin(); i != multiV.end(); i++)
    {
        instVertexNodes.push_back(new VertexNode(*(*i)));
        problem.AddNode(instVertexNodes.back());
    }
    instVertexNodes.back()->SetPreconditioner(preconditioners[0]);

    // instances rotations
    vector<RotationNode *> instRotationNodes;
    for (auto i = multiX.begin(); i != multiX.end(); i++)
    {
        instRotationNodes.push_back(new RotationNode(*(*i)));
        problem.AddNode(instRotationNodes.back());
    }
    instRotationNodes.back()->SetPreconditioner(preconditioners[1]);

    // instance barycentric coordinates
    vector<BarycentricNode *> instBarycentricNodes;
    vector<MeshWalker *> meshWalkers;

    for (int i = 0; i < multiU.size(); i++)
    {
        MeshWalker * meshWalker = new MeshWalker(mesh, *multiV[i]);
        meshWalkers.push_back(meshWalker);

        instBarycentricNodes.push_back(new BarycentricNode(*multiU[i], *multiL[i], *meshWalker));
        problem.AddNode(instBarycentricNodes.back());
    }
    instBarycentricNodes.back()->SetPreconditioner(preconditioners[2]);

    // add energies

    // dual arap
    for (int i = 0; i < instVertexNodes.size(); i++)
    {
        problem.AddEnergy(new DualARAPEnergy(*coreVertexNode, *instRotationNodes[i], *instVertexNodes[i], 
                          mesh, sqrt(lambdas[0]), uniformWeights));
    }

    // silhouette
    for (int i = 0; i < instVertexNodes.size(); i++)
    {
        problem.AddEnergy(new SilhouetteProjectionEnergy(*instVertexNodes[i], *instBarycentricNodes[i],
            *multiS[i], mesh, sqrt(lambdas[1]), narrowBand));

        problem.AddEnergy(new SilhouetteNormalEnergy(*instVertexNodes[i], *instBarycentricNodes[i],
            *multiSN[i], mesh, sqrt(lambdas[2]), narrowBand));
    }

    // laplacian
    problem.AddEnergy(new LaplacianEnergy(*coreVertexNode, mesh, sqrt(lambdas[3])));

    // minimise
    int ret = problem.Minimise(*options);

    // dealloc 
    dealloc_vector(meshWalkers);
    dealloc_vector(multiX);
    dealloc_vector(multiV);
    dealloc_vector(multiU);
    dealloc_vector(multiL);
    dealloc_vector(multiS);
    dealloc_vector(multiSN);

    return ret;
}

#endif
