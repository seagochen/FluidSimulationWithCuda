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
* <Last>		Dec 18, 2013
* <File>        parallelAux.cpp
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL\glew.h>
#include <GL\freeglut.h>

#include <SGE\SGUtils.h>
#include "cfd.h"

using namespace sge;

/*
-----------------------------------------------------------------------------------------------------------
* @function InitSimGrid
* @author   Orlando Chen
* @date     Dec 17, 2013
* @input    float *gridDensity, float *gridU, float *gridV, float *gridW
* @return   NULL
* @bref       
-----------------------------------------------------------------------------------------------------------
*/
void InitSimGrid ( float *gridDensity, float *gridU, float *gridV, float *gridW )
{
	int half = Grids_X / 2;
	int i = half - 10;  int maxi = half + 10;
	int k = half - 10;  int maxk = half + 10;

	for ( ; i < maxi; i++ )
	{
		for ( ; k < maxk; k++ )
		{
			int x = i - half;
			int y = k - half;
			float r = sqrt ( x * x + y * y );
			
			if ( r >= 0 && r <= 5 )
			{
				for ( int j = 0; j < 5; j++ )
				{
					gridDensity [ Index (i, j, k) ] = SOURCE;
					gridV [ Index (i, j, k) ] = 1.f;

					if ( i < half )
					{
						gridU [ Index (i, j, k) ] = - 0.1f;
						gridW [ Index (i, j, k) ] = - 0.1f;
					}
					else
					{
						gridU [ Index (i, j, k) ] = 0.1f;
						gridW [ Index (i, j, k) ] = 0.1f;
					}
				}
			}
		}
	}
};

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

	cudaFree ( dataset_dev );
	if ( dataset ) SAFE_FREE_PTR ( dataset );
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
		dataset [ i ] = 0;
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
		cudaCheckErrors ( "cudaSetDevices" );

	// Allocate memory on Host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static float *ptr;
		ptr = ( float * ) malloc ( SIM_SIZE*sizeof ( float ) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
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
			cudaCheckErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	// Allocate space for volumetric data set
	dataset = (unsigned char *) malloc ( SIM_SIZE * sizeof (unsigned char) );
	if ( cudaMalloc ( (void**)&dataset_dev, SIM_SIZE * sizeof (unsigned char) ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed" );
		return SG_RUNTIME_FALSE;
	}

	// Finally
	return SG_RUNTIME_OK;
}