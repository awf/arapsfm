/* axis_angle.h */
#ifndef __AXIS_ANGLE_H__
#define __AXIS_ANGLE_H__

#include "Math/static_linear.h"
#include "Geometry/quaternion.h"

// Functions for manipulating axis-angle representations of rotations directory

// axScale_Unsafe
template <typename Elem>
inline
void axScale_Unsafe(const Elem s, const Elem * x, Elem * y)
{
    scaleVector_Static<Elem, 3>(s, x, y);
}

// axAdd_Unsafe
template <typename Elem>
inline
void axAdd_Unsafe(const Elem * a, const Elem * b, Elem * c)
{
    Elem qa[4], qb[4], qr[4];
    quat_Unsafe(a, qa);
    quat_Unsafe(b, qb);

    quatMultiply_Unsafe(qa, qb, qr);

    quatInv_Unsafe(qr, c);
}

// axMakeInterpolated_Unsafe
template <typename Elem>
inline
void axMakeInterpolated_Unsafe(const Elem a, const Elem * v, const Elem b, const Elem * w, Elem * z)
{
    Elem av[3], bw[3];
    axScale_Unsafe(a, v, av);
    axScale_Unsafe(b, w, bw);
    axAdd_Unsafe(av, bw, z);
}

#endif

