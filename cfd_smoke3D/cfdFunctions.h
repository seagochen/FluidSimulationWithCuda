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
* <First>       Nov 26, 2013
* <Last>		Nov 26, 2013
* <File>        cfdFunctions.h
*/

#ifndef __cfd_functions_h_
#define __cfd_functions_h_

extern void DensitySolver ( float *grid, float *grid0, float *u, float *v, float *w );

extern void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 );

extern void DrawDensity ( void );

extern void DrawVelocity ( void );

extern void cudaAddSource (
	float *den_out, float *u_out, float *v_out, float *w_out, dim3 *gridDim, dim3 *blockDim );

extern void cudaVelAdvect ( 
	float *grid_out, float const *grid_in, int boundary,
	float const *u_in, float const *v_in, float const *w_in, dim3 *gridDim, dim3 *blockDim );

extern void cudaAnnihilation ( float *grid_out, dim3 *gridDim, dim3 *blockDim );

extern void cudaDiffuse ( 
	float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim );

extern void cudaViscosity (
	float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim );

extern void cudaDensAdvect ( 
	float *den_out, float const *dens_in, int boundary, 
	float const *u_in, float const *v_in, float const *w_in, dim3 *gridDim, dim3 *blockDim );

extern void cudaProjectField ( 
	float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in, dim3 *gridDim, dim3 *blockDim );

extern void cudaZeroData ( float *grid_out, dim3 *gridDim, dim3 *blockDim );

#endif