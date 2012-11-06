/* quaternion.h */
#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#include <cmath>
#include <cassert>
#include <algorithm>
#include "Math/static_linear.h"

// Functions for manipulating and evaluating quaternions from a vector of 3
// elements (axis-angle representation)

// General

// sinc = sin(t) / t 
// half_sinc_half_t = sin(0.5) / t

// half_sinc_half_t
template <typename Elem>
inline
Elem half_sinc_half_t(Elem t)
{
    assert(t >= 0);

    if (t < 1e-4)
        return 0.5;

    return sin(0.5*t) / t;
}

// half_sinc_half_t_p is the derivative w.r.t t of half_sinc_half_t

// half_sinc_half_t_p
template <typename Elem>
inline
Elem half_sinc_half_t_p(Elem t)
{
    assert(t >= 0);

    if (t < 1e-4)
        return 0;

    return 0.5*cos(0.5*t)/t - sin(0.5*t)/(t*t);
}

// half_sinc_half_t_p_div_t
template <typename Elem>
inline
Elem half_sinc_half_t_p_div_t(Elem t)
{
    assert(t >= 0);

    if (t < 1e-3)
        return -1./(24.);

    return half_sinc_half_t_p(t) / t;
}

// Evaluation functions (UNSAFE)

// quat_Unsafe
template <typename Elem>
inline
void quat_Unsafe(const Elem * x, Elem * q)
{
    Elem t = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    Elem a = half_sinc_half_t(t);

    q[0] = a*x[0];
    q[1] = a*x[1];
    q[2] = a*x[2];
    q[3] = cos(0.5*t);
}

// quatDx_Unsafe
template <typename Elem>
inline
void quatDx_Unsafe(const Elem * x, Elem * D)
{
    Elem t = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    Elem f = half_sinc_half_t(t);
    Elem g = half_sinc_half_t_p_div_t(t);

    D[0] = f + g*x[0]*x[0];
    D[1] = g*x[0]*x[1];
    D[2] = g*x[0]*x[2];

    D[3] = D[1];            // g*x[1]*x[0];
    D[4] = f + g*x[1]*x[1];
    D[5] = g*x[1]*x[2];

    D[6] = D[2];            // g*x[2]*x[0];
    D[7] = D[5];            // g*x[2]*x[1];
    D[8] = f + g*x[2]*x[2];

    D[9] = -0.5*f*x[0];
    D[10] = -0.5*f*x[1];
    D[11] = -0.5*f*x[2];
}

// rotationMatrix_Unsafe
template <typename Elem>
inline
void rotationMatrix_Unsafe(const Elem * q, Elem * R)
{
    R[0] = 1 - 2*q[1]*q[1] - 2*q[2]*q[2];
    R[1] = 2*q[0]*q[1] - 2*q[2]*q[3];
    R[2] = 2*q[0]*q[2] + 2*q[1]*q[3];
    R[3] = 2*q[0]*q[1] + 2*q[2]*q[3];
    R[4] = 1 - 2*q[0]*q[0] - 2*q[2]*q[2];
    R[5] = 2*q[1]*q[2] - 2*q[0]*q[3];
    R[6] = 2*q[0]*q[2] - 2*q[1]*q[3];
    R[7] = 2*q[1]*q[2] + 2*q[0]*q[3];
    R[8] = 1 - 2*q[0]*q[0] - 2*q[1]*q[1];
}

// quatMultiply_Unsafe
template <typename Elem>
inline
void quatMultiply_Unsafe(const Elem * p, const Elem * q, Elem * r)
{
    r[0] = p[3]*q[0] + p[0]*q[3] + p[1]*q[2] - p[2]*q[1];
    r[1] = p[3]*q[1] - p[0]*q[2] + p[1]*q[3] + p[2]*q[0];
    r[2] = p[3]*q[2] + p[0]*q[1] - p[1]*q[0] + p[2]*q[3];
    r[3] = p[3]*q[3] - p[0]*q[0] - p[1]*q[1] - p[2]*q[2];
}

// quatMultiply_dp_Unsafe
template <typename Elem>
inline
void quatMultiply_dp_Unsafe(const Elem * q, Elem * Dp)
{
    Dp[0] = q[3];
    Dp[1] = q[2];
    Dp[2] = -q[1];
    Dp[3] = q[0];
    Dp[4] = -q[2];
    Dp[5] = q[3];
    Dp[6] = q[0];
    Dp[7] = q[1];
    Dp[8] = q[1];
    Dp[9] = -q[0];
    Dp[10] = q[3];
    Dp[11] = q[2];
    Dp[12] = -q[0];
    Dp[13] = -q[1];
    Dp[14] = -q[2];
    Dp[15] = q[3];
}

// quatMultiply_dq_Unsafe
template <typename Elem>
inline
void quatMultiply_dq_Unsafe(const Elem * p, Elem * Dq)
{
    Dq[0] = p[3];
    Dq[1] = -p[2];
    Dq[2] = p[1];
    Dq[3] = p[0];
    Dq[4] = p[2];
    Dq[5] = p[3];
    Dq[6] = -p[0];
    Dq[7] = p[1];
    Dq[8] = -p[1];
    Dq[9] = p[0];
    Dq[10] = p[3];
    Dq[11] = p[2];
    Dq[12] = -p[0];
    Dq[13] = -p[1];
    Dq[14] = -p[2];
    Dq[15] = p[3];
}

// quatInv_Unsafe
template <typename Elem>
inline
void quatInv_Unsafe(const Elem * q, Elem * x)
{
    Elem t = 2.0 * atan2(sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]), q[3]);
    // t in [-pi, pi]

    Elem a = 1.0 / half_sinc_half_t(t);
    x[0] = q[0] * a;
    x[1] = q[1] * a;
    x[2] = q[2] * a;
}

// quatInvDq_Unsafe
template <typename Elem>
inline
void quatInvDq_Unsafe(const Elem * q, Elem * Dq)
{
    Elem m = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);

    if (m < 1e-6)
    {
        std::fill(Dq, Dq+12, 0.);
        Dq[0] = 2.0;
        Dq[5] = 2.0;
        Dq[10] = 2.0;
        return;
    }

    Elem dmdq[3] = {q[0] / m, q[1] / m, q[2] / m};

    Elem t = 2.0 * atan2(m, q[3]);
    Elem dtdm = 2.0 * q[3] / (m * m + q[3] * q[3]);
    Elem dtdq3 = -2.0 * m / (m * m + q[3] * q[3]);
    Elem dtdq[4] = {dtdm * dmdq[0],
                    dtdm * dmdq[1],
                    dtdm * dmdq[2],
                    dtdq3};

    Elem a = half_sinc_half_t(t);
    Elem a2 = a * a;
    Elem dadt = half_sinc_half_t_p(t);
    Elem dadq[4] = {dadt * dtdq[0],
                    dadt * dtdq[1],
                    dadt * dtdq[2],
                    dadt * dtdq[3]};

    Dq[0] = (-q[0] * dadq[0]) / a2 + 1.0 / a;
    Dq[1] = (-q[0] * dadq[1]) / a2;
    Dq[2] = (-q[0] * dadq[2]) / a2;
    Dq[3] = (-q[0] * dadq[3]) / a2;

    Dq[4] = (-q[1] * dadq[0]) / a2;
    Dq[5] = (-q[1] * dadq[1]) / a2 + 1.0 / a;
    Dq[6] = (-q[1] * dadq[2]) / a2;
    Dq[7] = (-q[1] * dadq[3]) / a2;

    Dq[8]  = (-q[2] * dadq[0]) / a2;
    Dq[9]  = (-q[2] * dadq[1]) / a2;
    Dq[10] = (-q[2] * dadq[2]) / a2 + 1.0 / a;
    Dq[11] = (-q[2] * dadq[3]) / a2;
}

#endif

