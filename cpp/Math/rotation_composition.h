#ifndef __ROTATION_COMPOSITION_H__
#define __ROTATION_COMPOSITION_H__

#include "Math/v3d_linear.h"
using namespace V3D;

#include "Math/static_linear.h"
#include "Solve/node.h"

#include "Geometry/quaternion.h"
#include "Geometry/axis_angle.h"

#include <cmath>
#include <vector>
#include <algorithm>
using namespace std;

// Rotation
class RotationComposition
{
public:
    RotationComposition(const vector<const RotationNode *> & Xb,
                        const vector<const CoefficientsNode *> & y)
        : _Xb(Xb), _y(y)
    {}

    void Rotation_Unsafe(int k, double * x, double * q = nullptr) const
    {
        fillVector_Static<double, 3>(0., x);

        for (int n = 0; n < _Xb.size(); n++)
        {
            double xn[3];
            copyVector_Static<double, 3>(_Xb[n]->GetRotation(k), xn);
            axScale_Unsafe(_y[n]->GetCoefficient(k), xn, xn);
            axAdd_Unsafe(xn, x, x);
        }

        if (q != nullptr)
            quat_Unsafe(x, q);
    }

    void Jacobian_Unsafe(int i, int k, bool isRotation, double * J) const
    {
        double x[3] = {0.};
        for (int r = 0; r < i; r++)
        {
            double xr[3];
            axScale_Unsafe(_y[r]->GetCoefficient(k), 
                           _Xb[r]->GetRotation(k),
                           xr);
            axAdd_Unsafe(xr, x, x);
        }

        double xi[3];
        axScale_Unsafe(_y[i]->GetCoefficient(k),
                       _Xb[i]->GetRotation(k),
                       xi);

        double Jq[9] = {0.};
        axAdd_da_Unsafe(xi, x, Jq);

        axAdd_Unsafe(xi, x, x);

        for (int l = i + 1; l < _Xb.size(); l++)
        {
            double xl[3];
            axScale_Unsafe(_y[l]->GetCoefficient(k),
                           _Xb[l]->GetRotation(k),
                           xl);

            double Jp[9];
            axAdd_db_Unsafe(xl, x, Jp);

            double Jr[9];
            multiply_A_B_Static<double, 3, 3, 3>(Jp, Jq, Jr);
            copyVector_Static<double, 9>(Jr, Jq);

            axAdd_Unsafe(xl, x, x);
        }

        if (isRotation)
        {
            scaleVectorIP_Static<double, 9>(_y[i]->GetCoefficient(k), Jq);
            copyVector_Static<double, 9>(Jq, J);
        }
        else
        {
            multiply_A_v_Static<double, 3, 3>(Jq, _Xb[i]->GetRotation(k), J);
        }
    }

private:
    const vector<const RotationNode *> & _Xb;
    const vector<const CoefficientsNode *> _y;
};

#endif
