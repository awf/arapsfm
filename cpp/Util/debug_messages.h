#ifndef __DEBUG_MESSAGES_H__
#define __DEBUG_MESSAGES_H__

// Convenience
#include <iostream>

// printVector_Static
template <typename Elem>
void printVector_Static(Elem * v, int n)
{
    std::cout << "[ ";
    for (int i=0; i<n; i++)
    {
        std::cout << v[i] << " ";
    }
    std::cout << "]";
}

#ifndef NO_HELPER
#define PRINT_VECTOR(n, X) do { std::cout << "> " << #X << ": "; printVector_Static(X, n); std::cout << std::endl; } while(false)
#define PRINT_VARIABLE(X) do { std::cout << "> " << #X << ": " << X << std::endl; } while (false)
#else
#define PRINT_VECTOR(n, X) do {} while (false)
#define PRINT_VARIABLE(X) do {} while (false)
#endif

#define PRINT_VECTOR2(X) PRINT_VECTOR(2, X)
#define PRINT_VECTOR3(X) PRINT_VECTOR(3, X)

#endif
