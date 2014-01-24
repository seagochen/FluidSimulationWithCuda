/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Jan 08, 2014
* <File>        FunctionHelperDynamic.cpp
*/

#ifndef __function_helper_dynamic_h_
#define __function_helper_dynamic_h_

#include <stdio.h>
#include <stdarg.h> 
#include <string>
#include <memory>
#include <cuda_runtime.h>

extern std::string string_fmt ( const std::string fmt_str, ... );
extern void cudaCheckErrors ( const char* msg, const char *file, const int line );

#endif