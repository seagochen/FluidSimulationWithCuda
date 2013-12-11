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
* <File>        cfdHeader.h
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
   etc.
  -----------------------------------------------------------------------------------------------------------
*/

#include "cudaNewHelper.h"
#include "visualFramework.h"
#include "GLSLHelper.h"
#include "resource.h"

/*
  -----------------------------------------------------------------------------------------------------------
   Parameters for Computational Fluid Dynamics
  -----------------------------------------------------------------------------------------------------------
*/

#define DELTA_TIME           0.1f   /* -------------------------------------- 0.1 second */
#define DIFFUSION            0.1f   /* -------------------------------------- diffusion */
#define VISCOSITY            0.0f   /* -------------------------------------- viscosity */
#define FORCE                5.0f   /* -------------------------------------- external force */
#define SOURCE               100.f  /* -------------------------------------- indensity */

/*
  -----------------------------------------------------------------------------------------------------------
   Global Data
  -----------------------------------------------------------------------------------------------------------
*/

#ifdef __launch_main_cpp_

std::vector <float*> dev_list;         /* ------------------------ a vector for storing device ptr */
std::vector <float*> host_list;        /* ------------------------ a vector for storing host ptr */
std::vector <float*> buffer_host_list; /* ------------------------ a vector for storing buffer ptr on host */
std::vector <float*> buffer_dev_list;  /* ------------------------ a vector for storing buffer ptr on device */
sge::FileManager    Logfile;           /* ------------------------ SGE file manager, for runtime message */

// shader and texture handle
GLuint m_tffTexObj;    /* 1-D ray casting texture */
GLuint m_bfTexObj;     /* 2-D backface texture */
GLuint m_volTexObj;    /* 3-D volume data texture */
GLuint m_rcVertHandle; /* vertex shader of rc */
GLuint m_rcFragHandle; /* fragment shader of rc */
GLuint m_bfVertHandle; /* vertex shader of bf */
GLuint m_bfFragHandle; /* fragment shader of bf */

GLuint m_programHandle;
GLuint m_frameBuffer;

sge::_mouse        *m_mouse;
sge::_fps          *m_fps;
sge::_viewMatrix   *m_view;

GLint  m_width, m_height;
bool   m_density;
GLuint m_cluster;
GLint  m_angle;

#else

extern std::vector <float*> dev_list;         /* ----------------- declaration of device vector */
extern std::vector <float*> host_list;        /* ----------------- declaration of host vector */
extern std::vector <float*> buffer_host_list; /* ----------------- declaration of host buffer */
extern std::vector <float*> buffer_dev_list;  /* ----------------- declaration of device buffer*/
extern sge::FileManager    Logfile;           /* ----------------- declaration of SGE file manager */

extern GLuint m_tffTexObj;    /* 1-D ray casting texture */
extern GLuint m_bfTexObj;     /* 2-D backface texture */
extern GLuint m_volTexObj;    /* 3-D volume data texture */
extern GLuint m_rcVertHandle; /* vertex shader of rc */
extern GLuint m_rcFragHandle; /* fragment shader of rc */
extern GLuint m_bfVertHandle; /* vertex shader of bf */
extern GLuint m_bfFragHandle; /* fragment shader of bf */

extern GLuint m_programHandle;
extern GLuint m_frameBuffer;

extern sge::_mouse        *m_mouse;
extern sge::_fps          *m_fps;
extern sge::_viewMatrix   *m_view;

extern GLint  m_width, m_height;
extern bool   m_density;
extern GLuint m_cluster;
extern GLint  m_angle;

#endif

/*
  -----------------------------------------------------------------------------------------------------------
   external functions
  -----------------------------------------------------------------------------------------------------------
*/

#include "cfdFunctions.h"

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

#define BufferHostListNum     4
#define host_display_u        buffer_host_list [ 0 ] /* ---------- rendering buffer for u */
#define host_display_v        buffer_host_list [ 1 ] /* ---------- rendering buffer for v */
#define host_display_w        buffer_host_list [ 2 ] /* ---------- rendering buffer for w */
#define host_display_den      buffer_host_list [ 3 ] /* ---------- rendering buffer for density */

#define BufferDeviceListNum   2
#define dev_display_temp2D1   buffer_dev_list [ 0 ] /* ----------- ray casting buffer, CUDA device */
#define dev_display_temp2D2   buffer_dev_list [ 1 ] /* ----------- ray casting buffer, CUDA device */


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

/*
  -----------------------------------------------------------------------------------------------------------
   etc.
  -----------------------------------------------------------------------------------------------------------
*/

extern void initShader ( void );
extern GLuint initVol3DTex ( const char* filename, GLuint w, GLuint h, GLuint d );
extern GLuint initFace2DTex ( GLuint bfTexWidth, GLuint bfTexHeight );
extern GLuint initTFF1DTex ( const char* filename );
extern sge::SGRUNTIMEMSG AllocateResourcePtrs ();
extern void FreeResourcePtrs ();
extern void ZeroData ();
extern void InitViewMatrix(), InitFont(), InitFPS(), InitMouseStatus(), Setup();
extern void RenderingFace ( GLenum cullFace );
extern void SetVolumeInfoUinforms ( void );
extern void CountFPS ( void );

#endif