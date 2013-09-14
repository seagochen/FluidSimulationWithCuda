/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <Date>        Sep 12, 2013
* <File>        Auxiliaries.h
*/

#ifndef _SEAGOSOFT_AUXILIARY_H_
#define _SEAGOSOFT_AUXILIARY_H_

#ifdef SAFE_FREE_PTR(ptr)
#undef SAFE_FREE_PTR(ptr)
#endif

#ifdef SAFE_DELT_PTR(ptr)
#undef SAFE_DELT_PTR(ptr)
#endif

#ifdef SAFE_DELT_ARR(ptr)
#undef SAFE_DELT_ARR(ptr)
#endif

#ifdef _DLL
#undef _DLL
#endif

#ifdef pterror(str)
#undef pterror(str)
#endif

#define SAFE_FREE_PTR(ptr)   { if( ptr ) free(ptr);  ptr = NULL; }   // rid ptr from calloc or malloc
#define SAFE_DELT_PTR(ptr)   { if( ptr ) delete ptr; ptr = NULL; }   // rid ptr, such as int *ptr
#define SAFE_DELT_ARR(ptr)   { if( ptr ) delete []ptr; ptr = NULL; } // rid ptr array

#ifdef _In_Dll_File
#define _DLL __declspec(dllexport)    // Headers included in dll source
#else
#define _DLL __declspec(dllimport)    // Headers included from external  
#endif  

#ifdef _In_Dll_File
#include <stdio.h>
#endif

#ifdef SG_OK
#undef SG_OK
#endif

#ifdef SG_FAIL
#undef SG_FAIL
#endif

#ifdef SG_ERROR
#undef SG_ERROR
#endif

#define SG_OK     1
#define SG_FAIL   0
#define SG_ERROR -1

#ifdef _In_Dll_File
#include <GL\glew.h>
#include <GL\glut.h>
#endif

/// Some Functions ///
#define ErrorMSG(str) { printf("Error> %s, check your code at line %d of %s in file %s\n", \
	str, __LINE__, __FUNCTIONW__, __FILE__);}

namespace sge
{
	class float3
	{

	};
};

#endif