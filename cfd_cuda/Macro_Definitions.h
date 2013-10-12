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
* <First>       Oct 7, 2013
* <Last>		Oct 7, 2013
* <File>        Macro_Definitions.h
*/

#ifndef _CUSTOM_MACRO_DEFINITIONS_H_
#define _CUSTOM_MACRO_DEFINITIONS_H_

////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Normal Definitions
  ----------------------------------------------------------------------
*/

#define True  1
#define False 0

/*
  ----------------------------------------------------------------------
   Definition for Variables of Computational Fluid Dynamics
  ----------------------------------------------------------------------
*/

#define GRIDS_WITHOUT_GHOST  64       // grids number without ghost grids
#define ENTIRE_GRIDS_NUMBER  66       // grids number contains ghost grids
#define DELTA_TIME           0.1f     // 0.1 second
#define DIFFUSION            0.0f     // diffusion rate
#define VISCOSITY            0.0f     // viscosity rate
#define FORCE                5.0f     // external force rate
#define SOURCE               100.0f   // to given a density with 100 percent
#define WINDOWSX             512      // application window size, width
#define WINDOWSY             512      // application window size, height

/*
  ----------------------------------------------------------------------
   Definition for Visualization
  ----------------------------------------------------------------------
*/

#define BYTES_PER_TEXEL 3

///
////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Definition of Switch
  ----------------------------------------------------------------------
*/

#define USING_GPU_FOR_CFD  False
#define GPU_SWITCH         USING_GPU_FOR_CFD

///
////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Function Definitions
  ----------------------------------------------------------------------
*/

#define PrintStatus(str) {system("cls"); printf("%s");}

#define Index(i,j) ((j)*ENTIRE_GRIDS_NUMBER + i)

template <class T> void SWAP ( T& a, T& b )
{
  T c(a); a=b; b=c;
}

/*
  ----------------------------------------------------------------------
   external functions on CPU
  ----------------------------------------------------------------------
*/


void dens_step ( float * grid, float * grid0, float * u, float * v );

void vel_step ( float * u, float * v, float * u0, float * v0 );


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


void free_data ( void );

void clear_data ( void );

int allocate_data ( void );


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

void draw_velocity ( void );

void draw_density ( void );


/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

void get_from_UI ( float * d, float * u, float * v );

/*
  ----------------------------------------------------------------------
   Mainroutine
  ----------------------------------------------------------------------
*/

#include <SGE\SGUtils.h>

void key_func ( sge::SG_KEYS key, sge::SG_KEY_STATUS status );

void mouse_func ( sge::SG_MOUSE mouse, unsigned x, unsigned y );

void reshape_func ( unsigned width, unsigned height );

void display_func ( void );

void idle_func( void );

void dest_func( void );


#endif