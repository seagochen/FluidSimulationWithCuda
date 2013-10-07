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
* <File>        CFD_FuncsPrototype.h
*/


#ifndef _SEAGOSOFT_CUSTOM_FUNCTION_PROTOTYPES_H_
#define _SEAGOSOFT_CUSTOM_FUNCTION_PROTOTYPES_H_

#include <stdlib.h>
#include <stdio.h>
#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include "Macro_Funcs.h"

/*
  ----------------------------------------------------------------------
   external functions
  ----------------------------------------------------------------------
*/


extern void dens_step ( int GridSize, float * x, float * x0, float * u, float * v, float diff, float dt );

extern void vel_step ( int GridSize, float * u, float * v, float * u0, float * v0, float visc, float dt );


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

using sge::SG_MOUSE;
using sge::SG_KEYS;
using sge::SG_KEY_STATUS;

void key_func ( SG_KEYS key, SG_KEY_STATUS status );

void mouse_func ( SG_MOUSE mouse, unsigned x, unsigned y );

void reshape_func ( unsigned width, unsigned height );

void display_func ( void );

void idle_func( void );

void dest_func( void );


#endif