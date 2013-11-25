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
* <First>       Nov 18, 2013
* <Last>		Nov 25, 2013
* <File>        CFDSolver.cpp
*/

#ifndef __velocity_solver_cpp_
#define __velocity_solver_cpp_

#include "cfdHeader.h"

using namespace std;


/*
-----------------------------------------------------------------------------------------------------------
* @function VelocitySolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    u, v, w, u0, v0, w0
* @return   NULL
* @bref     Calculate the advection of flow, and update the velocity on each cell
*      
-----------------------------------------------------------------------------------------------------------
*/
void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	cudaAddSource ( NULL, dev_u, dev_v, dev_w, &gridDim, &blockDim );  
	swap ( dev_u0, dev_u ); cudaViscosity ( dev_u, dev_u0, 1, &gridDim, &blockDim );
	swap ( dev_v0, dev_v ); cudaViscosity ( dev_v, dev_v0, 2, &gridDim, &blockDim );

	cudaProjectField ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, &gridDim, &blockDim );
	swap ( dev_u0, dev_u ); swap ( dev_v0, dev_v );

	cudaVelAdvect ( dev_u, dev_u0, 1, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	cudaVelAdvect ( dev_v, dev_v0, 2, dev_u0, dev_v0, dev_w0, &gridDim, &blockDim );
	
	cudaProjectField ( dev_grid, dev_grid0, dev_u, dev_v, dev_w, &gridDim, &blockDim );

    // Copy output vector from GPU buffer to host memory.	
	if ( cudaMemcpy ( u, dev_u, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( v, dev_v, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );

	if ( cudaMemcpy ( w, dev_w, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function DensitySolver
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    float *grid, float *grid0, float *u, float *v, float *w
* @return   NULL
* @bref     Calculate the advection of flow, and update the density on each cell     
-----------------------------------------------------------------------------------------------------------
*/
void DensitySolver ( float *dens, float *dens0, float *u, float *v, float *w )
{
	// Define the computing unit size
	cudaDeviceDim3D ( );

	cudaAddSource ( dev_den, NULL, NULL, NULL, &gridDim, &blockDim );  swap( dev_den, dev_den0 );
	cudaDiffuse ( dev_den, dev_den0, 0, &gridDim, &blockDim ); swap( dev_den, dev_den0 );
    cudaDensAdvect ( dev_den, dev_den0, 0, dev_u, dev_v, dev_w, &gridDim, &blockDim );
	cudaAnnihilation ( dev_den,  &gridDim, &blockDim );
	cudaAnnihilation ( dev_den0,  &gridDim, &blockDim );

    // Copy output vector from GPU buffer to host memory.
	if ( cudaMemcpy ( dens, dev_den, SIM_SIZE * sizeof(float), cudaMemcpyDeviceToHost ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaMemcpy was failed" );
}

#endif