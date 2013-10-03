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

/// Undefine those macro definitions if already defined ///
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


/// Safe release ptr ///
#define SAFE_FREE_PTR(ptr)   { if( ptr ) free(ptr);  ptr = NULL; }   // rid ptr from calloc or malloc
#define SAFE_DELT_PTR(ptr)   { if( ptr ) delete ptr; ptr = NULL; }   // rid ptr, such as int *ptr
#define SAFE_DELT_ARR(ptr)   { if( ptr ) delete []ptr; ptr = NULL; } // rid ptr array


/// DLL ///
#ifdef _In_Dll_File
#define _DLL __declspec(dllexport)    // Headers included in dll source
#else
#define _DLL __declspec(dllimport)    // Headers included from external  
#endif  

#ifdef _In_Dll_File
#include <stdio.h>
#endif

/// SG runtime message ///
namespace sge
{
	enum SGRUNTIMEMSG
	{
		/// Runtime OK
		SG_RUNTIME_OK,		            // return SG_RUNTIME_OK when result is ok
		SG_RUNTIME_FALSE,               // return SG_RUNTIME_FALSE when result is false

		/// Runtime Exceptions
		SG_NULL_POINTER_EXCEP,          // return SG_NULL_POINTER when a null pointer passed in
		SG_NO_TARGET_FOUND_EXCEP,       // return SG_NO_TARGET_FOUND_EXCEP when no file found
		SG_READ_TARGET_FAILED_EXCEP,    // return SG_READ_TARGET_FAILED_EXCEP when failed to read a file
		SG_WRITE_TARGET_FAILED_EXCEP,   // return SG_WRITE_TARGET_FAILED_EXCEP when failed to write a file
		SG_CALL_EXTFUNC_FAILED_EXCEP,   // return SG_CALL_EXTFUNC_FAILED_EXCEP when call an external function failed

		/// Logical error
		SG_ZERO_DENOMINATOR_ERROR,      // return SG_ZERO_DENOMINATOR_ERROR when denominator is zero
		SG_OUT_OF_RANGE_ERROR,          // return SG_OUT_OF_RANGE_ERROR when result is out of range
	};
};

#ifdef _In_Dll_File
#include <GL\glew.h>
#include <GL\glut.h>
#endif

/// Some Functions ///
#define ErrorMSG(str) { printf("Error> %s, check your code at line %d in file %s\n", \
	str, __LINE__, __FILE__);}


/// Some macro definitions ///
#define elif else if

#endif