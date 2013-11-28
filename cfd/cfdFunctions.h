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

/*
  -----------------------------------------------------------------------------------------------------------
   Solvers
  -----------------------------------------------------------------------------------------------------------
*/

/*
-----------------------------------------------------------------------------------------------------------
* @function DensitySolver
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *dens, float *dens0, float *u, float *v, float *w
* @return   NULL
* @bref     Add Some particles in the velocity field and calculate how it effects on these particles
-----------------------------------------------------------------------------------------------------------
*/
extern void DensitySolver ( float *grid, float *grid0, float *u, float *v, float *w );


/*
-----------------------------------------------------------------------------------------------------------
* @function VelocitySolver
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *u, float *v, float *w, float *u0, float *v0, float *w0
* @return   NULL
* @bref     To solve the velocity field of fluid
-----------------------------------------------------------------------------------------------------------
*/
extern void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 );


/*
  -----------------------------------------------------------------------------------------------------------
   Drawers
  -----------------------------------------------------------------------------------------------------------
*/

/*
-----------------------------------------------------------------------------------------------------------
* @function DrawDensity
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    NULL
* @return   NULL
* @bref     To display the result of density
-----------------------------------------------------------------------------------------------------------
*/
extern void DrawDensity ( void );


/*
-----------------------------------------------------------------------------------------------------------
* @function DrawVelocity
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    NULL
* @return   NULL
* @bref     To display the result of velocity field
-----------------------------------------------------------------------------------------------------------
*/
extern void DrawVelocity ( void );


/*
  -----------------------------------------------------------------------------------------------------------
   CUDA Kernels
  -----------------------------------------------------------------------------------------------------------
*/

/*
-----------------------------------------------------------------------------------------------------------
* @function cudaAddSource
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float *u_out, float *v_out, float *w_out, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Add new velocity and density to the field
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaAddSource 
	( float *den_out, float *u_out, float *v_out, float *w_out, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaVelAdvect
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float const *dens_in, int boundary,
* --------- float const *u_in, float const *v_in, float const *w_in
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update the status of velocity field
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaVelAdvect 
	( float *grid_out, float const *grid_in, int boundary,
	float const *u_in, float const *v_in, float const *w_in, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaDensAdvect
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *den_out, float const *dens_in, int boundary,
* --------- float const *u_in, float const *v_in, float const *w_in
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update the status of density
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaDensAdvect
	( float *den_out, float const *dens_in, int boundary, 
	float const *u_in, float const *v_in, float const *w_in, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaSetBoundary
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Boundary condition
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaSetBoundary ( float *grid_out, int boundary, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaDiffuse
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Diffusion
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaDiffuse 
	( float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaViscosity
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Viscosity
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaViscosity
	( float *grid_out, float const *grid_in, int boundary, dim3 *gridDim, dim3 *blockDim );


/*
-----------------------------------------------------------------------------------------------------------
* @function cudaProjectField
* @author   Orlando Chen
* @date     Nov 26, 2013
* @input    float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in,
* --------- dim3 *gridDim, dim3 *blockDim
* @return   NULL
* @bref     Update velocity field
-----------------------------------------------------------------------------------------------------------
*/
extern void cudaProjectField 
	( float *grad_in, float *proj_out, float *u_in, float *v_in, float *w_in, dim3 *gridDim, dim3 *blockDim );

#endif