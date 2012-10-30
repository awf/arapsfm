/* test_move_linear.cpp */

// Compile with gcc 4.7.1 current directory:
// g++ test_move_linear.cpp -std=c++11 -I../../cpp/SSLM/ 

// Includes
#define DEBUG_MOVE_SEMANTICS
#include "Math/v3d_linear.h"
#include <vector>
#include <utility>
#include <iostream>
using namespace V3D;

// test
template <template<typename V> class T, typename U>
int test()
{
    std::vector<T<U>> vec;
    T<U> v;
    std::cout << "1" << std::endl;
    vec.push_back(std::move(v));
    std::cout << "2" << std::endl;
    vec.push_back(T<U>());
    std::cout << "3" << std::endl;
    vec.push_back(T<U>());
    std::cout << "4" << std::endl;
    vec.push_back(T<U>());

    return 0;
}

// main
int main()
{
    std::cout << "test<Matrix, int>:" << std::endl;
    test<Matrix, int>();

    std::cout << "test<Vector, int>:" << std::endl;
    test<Vector, int>();
}

