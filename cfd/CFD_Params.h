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
* <Date>        Oct 6, 2013
* <File>        CFD_Params.h
*/

#ifndef _SEAGOSOFT_CUSTOM_PARAMS_H_
#define _SEAGOSOFT_CUSTOM_PARAMS_H_

/* global variables */

int GridSize;
float dt, diff, visc;
float force, source;

float * u, * v, * u_prev, * v_prev;
float * dens, * dens_prev;

bool mouse_down[2];
int omx, omy, mx, my;

int win_x, win_y;


void CFD_Init_Params()
{
	GridSize = 64;
	dt = 0.1f;
	diff = 0.0f;
	visc = 0.0f;
	force = 5.0f;
	source = 100.0f;
	win_x = 256;
	win_y = 256;
};

#endif