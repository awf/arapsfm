#ifndef __TEST_PROBLEM_H__
#define __TEST_PROBLEM_H__

#include <Math/v3d_linear.h>
using namespace V3D;

#include "Solve/problem.h"
#include "Solve/node.h"

// test_problem
void test_problem()
{
    Matrix<double> V(100, 3);
    Matrix<double> V1(100, 3);
    Matrix<double> X(100, 3);

    Problem problem;
    problem.AddNode(new VertexNode(V));
    problem.AddNode(new VertexNode(V1));
    problem.AddNode(new RotationNode(X));
    problem.InitialiseParamDesc();

}

#endif
