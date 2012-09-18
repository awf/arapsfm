/* static_linear.h */
#ifndef __STATIC_LINEAR_H__
#define __STATIC_LINEAR_H__

#include <cmath>
#include <cassert>

// Useful defines
#define PI      (3.141592653589793)
#define TWO_PI  (6.283185307179586)
#define HALF_PI (1.570796326794896)

// Vector and matrix operations with statically set limits operating directly
// on memory

// multiply_A_v_Static
template <typename Elem, int rows, int cols>
inline
void multiply_A_v_Static(const Elem * A, const Elem * x, Elem * b)
{
    for (int i=0; i < rows; i++)
    {
        Elem acc = 0;
        for (int j=0; j < cols; j++)
            acc += A[i*cols + j]*x[j];
        b[i] = acc;
    }
}

// multiply_A_B_Static
template <typename Elem, int rows, int dim, int cols>
void multiply_A_B_Static(const Elem * A, const Elem * B, Elem * C)
{
    for (int i=0; i < rows; i++)
        for (int j=0; j < cols; j++)
        {
            Elem acc = 0;

            for (int l=0; l < dim; l++)
                acc += A[i*dim + l]*B[l*cols + j];

            C[i*cols + j] = acc;
        }
}

// multiplyVectors_Static
template <typename Elem, int n>
inline
void multiplyVectors_Static(const Elem * a, const Elem * b, Elem * c)
{
    for (int i=0; i < n; i++) c[i] = a[i] * b[i];
}

// subtractVectors_Static
template <typename Elem, int n>
inline
void subtractVectors_Static(const Elem * a, const Elem * b, Elem * c)
{
    for (int i=0; i < n; i++) c[i] = a[i] - b[i];
}

// addVectors_Static
template <typename Elem, int n>
inline
void addVectors_Static(const Elem * a, const Elem * b, Elem * c)
{
    for (int i=0; i < n; i++) c[i] = a[i] + b[i];
}

// scaleVectorIP_Static
template <typename Elem, int n>
inline
void scaleVectorIP_Static(const Elem a, Elem * b)
{
    for (int i=0; i < n; i++) b[i] *= a;
}

// scaleVector_Static
template <typename Elem, int n>
inline
void scaleVector_Static(const Elem a, const Elem * b, Elem * c)
{
    for (int i=0; i < n; i++) c[i] = b[i] * a;
}

// sqrNorm_L2_Static
template <typename Elem, int n>
inline
Elem sqrNorm_L2_Static(const Elem * a)
{
    Elem acc = 0;
    for (int i=0; i < n; i++) acc += a[i]*a[i];
    return acc;
}

// norm_L2_Static
template <typename Elem, int n>
inline
Elem norm_L2_Static(const Elem * a)
{
    return sqrt(sqrNorm_L2_Static<Elem, n>(a));
}

// normalizeVector_Static
template <typename Elem, int n>
inline
Elem normalizeVector_Static(Elem * a)
{
    Elem l = norm_L2_Static<Elem, n>(a);

    assert(l > 0.);

    scaleVectorIP_Static<Elem, n>(1.0 / l, a);

    return l;
}

// crossProduct_Static
template <typename Elem>
inline
void crossProduct_Static(const Elem * a, const Elem * b, Elem * c)
{
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

// innerProduct_Static
template <typename Elem, int n>
inline
Elem innerProduct_Static(const Elem * a, const Elem * b)
{
    Elem res = 0;
    for (int i=0; i < n; i++) res += a[i]*b[i];
    return res;
}

// makeInterpolatedVector_Static
template <typename Elem, int n>
inline
void makeInterpolatedVector_Static(const Elem a, const Elem * v, const Elem b, const Elem * w, Elem * z)
{
    for (int i=0; i < n; i++) z[i] = a*v[i] + b*w[i];
}

// fillVector_Static
template <typename Elem, int n>
inline
void fillVector_Static(Elem a, Elem * b)
{
    for (int i=0; i < n; i++) b[i] = a;
}

// makeTriInterpolatedVector_Static
template <typename Elem, int n>
inline
void makeTriInterpolatedVector_Static(const Elem * u, const Elem * a, const Elem * b, const Elem * c, Elem * z)
{
    for (int i=0; i < n; i++)
        z[i] = u[0]*a[i] + u[1]*b[i] + u[2]*c[i];
}

// clipVector_Static
template <typename Elem, int n>
inline
void clipVector_Static(const Elem a, const Elem b, Elem * v)
{
    for (int i=0; i < n; i++)
    {
        if (v[i] < a) v[i] = a;
        if (v[i] > b) v[i] = b;
    }
}

// sumVector_Static
template <typename Elem, int n>
inline
Elem sumVector_Static(const Elem * v)
{
    Elem acc = Elem(0);
    for (int i=0; i < n; i++) acc += v[i];
    return acc;
}

#endif

