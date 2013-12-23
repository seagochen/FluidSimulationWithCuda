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
* <First>       Oct 25, 2013
* <Last>		Nov 18, 2013
* <File>        cfd.h
*/

#ifndef __cfd_header_h_
#define __cfd_header_h_

/*
  -----------------------------------------------------------------------------------------------------------
   C Standard Headers
  -----------------------------------------------------------------------------------------------------------
*/

#include <stdio.h>
#include <stdlib.h>

/*
  -----------------------------------------------------------------------------------------------------------
   C++ STL Headers and System Headers
  -----------------------------------------------------------------------------------------------------------
*/

#include <iostream>
#include <vector>

/*
  -----------------------------------------------------------------------------------------------------------
   OpenGL Headers
  -----------------------------------------------------------------------------------------------------------
*/

#include <GL\glew.h>
#include <GL\glut.h>

/*
  -----------------------------------------------------------------------------------------------------------
   CUDA Headers
  -----------------------------------------------------------------------------------------------------------
*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
  -----------------------------------------------------------------------------------------------------------
   SGEngine Headers
  -----------------------------------------------------------------------------------------------------------
*/

#include <SGE\SGUtils.h>

/*
  -----------------------------------------------------------------------------------------------------------
   Parameters for Computational Fluid Dynamics
  -----------------------------------------------------------------------------------------------------------
*/

#include "cudabox.h"

#define DELTA_TIME           0.1f   /* -------------------------------------- 0.1 second */
#define DIFFUSION            0.1f   /* -------------------------------------- diffusion */
#define VISCOSITY            0.0f   /* -------------------------------------- viscosity */
#define FORCE                5.0f   /* -------------------------------------- external force */
#define SOURCE               150    /* -------------------------------------- indensity */
#define SIM_SIZE             Grids_X * Grids_X * Grids_X /* ----------- CFD dimension of grid */
/*
  -----------------------------------------------------------------------------------------------------------
   Global Data
  -----------------------------------------------------------------------------------------------------------
*/

#ifdef __main_cpp_

std::vector <float*> dev_list;         /* ------------------------ a vector for storing device ptr */
std::vector <float*> host_list;        /* ------------------------ a vector for storing host ptr */

unsigned char *dataset;
unsigned char *dataset_dev;

#else

extern std::vector <float*> dev_list;         /* ----------------- declaration of device vector */
extern std::vector <float*> host_list;        /* ----------------- declaration of host vector */

extern unsigned char *dataset;
extern unsigned char *dataset_dev;

#endif

/*
  -----------------------------------------------------------------------------------------------------------
   List
  -----------------------------------------------------------------------------------------------------------
*/

#define DevListNum    10
#define dev_u         dev_list [ 0 ] /* -------------------------- u of U, on CUDA */
#define dev_v         dev_list [ 1 ] /* -------------------------- v of U, on CUDA */
#define dev_w         dev_list [ 2 ] /* -------------------------- w of U, on CUDA */
#define dev_u0        dev_list [ 3 ] /* -------------------------- original u of U, on CUDA */
#define dev_v0        dev_list [ 4 ] /* -------------------------- original v of U, on CUDA */
#define dev_w0        dev_list [ 5 ] /* -------------------------- original w of U, on CUDA */
#define dev_den       dev_list [ 6 ] /* -------------------------- density, on CUDA */
#define dev_den0      dev_list [ 7 ] /* -------------------------- original density, on CUDA */
#define dev_grid      dev_list [ 8 ] /* -------------------------- temporary grid, on CUDA */
#define dev_grid0     dev_list [ 9 ] /* -------------------------- original temporary grid, on CUDA */

#define HostListNum   8
#define host_u        host_list [ 0 ] /* ------------------------- component u of flow U on x-axis (left to right) */
#define host_v        host_list [ 1 ] /* ------------------------- component v of flow U on y-axis (up to down) */
#define host_w        host_list [ 2 ] /* ------------------------- component w of flow U on z-axis (near to far) */
#define host_u0       host_list [ 3 ] /* ------------------------- original u */
#define host_v0       host_list [ 4 ] /* ------------------------- original v */
#define host_w0       host_list [ 5 ] /* ------------------------- original w */
#define host_den      host_list [ 6 ] /* ------------------------- scalar field of density */
#define host_den0     host_list [ 7 ] /* ------------------------- original density */


/*
  -----------------------------------------------------------------------------------------------------------
   Define something
  -----------------------------------------------------------------------------------------------------------
*/

#define eqt              ==            /* equal to */
#define and              &&            /* logical and */
#define or               ||            /* logical or */

#define gst_header       0             /* (ghost, halo) the header cell of grid */
#define sim_header       1             /* (actually) the second cell of grid */
#define gst_tailer       Grids_X - 1   /* (ghost, halo) the last cell of grid */
#define sim_tailer       Grids_X - 2   /* (actually) the second last cell of grid */

#define BeginSimArea() \
	if ( i >= sim_header and i <= sim_tailer ) \
	if ( j >= sim_header and j <= sim_tailer ) \
	if ( k >= sim_header and k <= sim_tailer ) {

#define EndSimArea() }

#endif