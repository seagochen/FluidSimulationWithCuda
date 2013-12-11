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
* <First>       Dec 12, 2013
* <Last>		Dec 12, 2013
* <File>        parallelAux.cpp
*/

#ifndef __parallel_aux_cpp_
#define __parallel_aux_cpp_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLT/glm/glm.hpp>
#include <GLT/glm/gtc/matrix_transform.hpp>
#include <GLT/glm/gtx/transform2.hpp>
#include <GLT/glm/gtc/type_ptr.hpp>

#include "cfdHeader.h"
#include "GLSLHelper.h"

using namespace sge;

/*
-----------------------------------------------------------------------------------------------------------
* @function FreeResourcePtrs
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Release resource      
-----------------------------------------------------------------------------------------------------------
*/
void FreeResourcePtrs ( void )
{
	// Release ptr of host
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	// Release ptr of CUDA
	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );

	// Release other resource
	for ( int i = 0; i < BufferHostListNum; i++ )
	{
		if ( buffer_host_list [ i ] ) SAFE_FREE_PTR ( buffer_host_list [ i ] );
	}
	buffer_host_list.empty ( );

	// And ...
	for ( int i = 0; i < BufferDeviceListNum; i++ )
	{
		cudaFree ( buffer_dev_list [ i ] );
	}
	buffer_dev_list.empty ( );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function ZeroData
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Zero the buffers
-----------------------------------------------------------------------------------------------------------
*/
void ZeroData ( void )
{
	for ( int i = 0; i < SIM_SIZE; i++ )
	{
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_u0 [ i ] = 0.f;
		host_v0 [ i ] = 0.f;
		host_w0 [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_den0 [ i ] = 0.f;
	}
}


/*
-----------------------------------------------------------------------------------------------------------
* @function AllocateResourcePtrs
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Allocate resource      
-----------------------------------------------------------------------------------------------------------
*/
SGRUNTIMEMSG AllocateResourcePtrs ( void )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaSetDevices" );

	// Allocate memory on Host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static float *ptr;
		ptr = ( float * ) malloc ( SIM_SIZE*sizeof ( float ) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND,
				"allocate data was failed, at line: %d of file %s", __LINE__, __FILE__ );
			return SG_RUNTIME_FALSE;
		}
	}

	// Allocate memory on GPU devices
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( ( void ** ) &ptr, SIM_SIZE * sizeof ( float ) ) != cudaSuccess )
		{
			cudaCheckRuntimeErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	// Allocate rendering buffers both on host and GPU devices
	// Host first
	for ( int i = 0; i < BufferHostListNum; i++ )
	{
		static float *ptr;
		ptr = ( float * ) malloc ( DIS_SIZE * sizeof ( float ) );
		buffer_host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! buffer_host_list [ i ] )
		{
			Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND,
				"allocate data was failed, at line: %d of file %s", __LINE__, __FILE__ );
			return SG_RUNTIME_FALSE;
		}
	}

	// Then GPU devices
	for ( int i = 0; i < BufferDeviceListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( ( void ** ) &ptr, DIS_SIZE * sizeof ( float ) ) != cudaSuccess )
		{
			cudaCheckRuntimeErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		buffer_dev_list.push_back(ptr);
	}

	// Finally
	return SG_RUNTIME_OK;
}

#endif