// -*- C++ -*-

#ifndef V3D_EXCEPTION_H
#define V3D_EXCEPTION_H

#include <string>
#include <sstream>
#include <iostream>
#include <assert.h>

#define verify(condition, message) do                      \
   {                                                       \
      if (!(condition)) {                                  \
         std::cout << "VERIFY FAILED: " << (message) << "\n"       \
              << "    " << __FILE__ << ", " << __LINE__ << "\n"; \
         assert(false);                                    \
         exit(0);                                          \
      }                                                    \
   } while(false);

#define throwV3DErrorHere(reason) {}

#endif
