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
* <First>       Oct 24, 2013
* <Last>		Oct 24, 2013
* <File>        cfd_rendering.cpp
*/

#ifndef __cfd_rendering_cpp_
#define __cfd_rendering_cpp_

#include <GL\freeglut.h>

#include "macro_def.h"

void draw_velocity(void)
{
	int i, j;
	float x, y, h;

	h = 1.0f/GridSize;

	glColor3f(0.0f, 0.0f, 1.0f);
	glLineWidth(1.0f);

	glBegin(GL_LINES);
	{
		for ( i=1 ; i<=GridSize ; i++ )
		{
			x = (i-0.5f)*h;
			for ( j=1 ; j<=GridSize ; j++ )
			{
				y = (j-0.5f)*h;
				glVertex2f(x, y);
				glVertex2f(x+u[Index(i,j)], y+v[Index(i,j)]);
			}
		}
	}
	glEnd();
}

void draw_density(void)
{
	int i, j;
	float x, y, h, d00, d01, d10, d11;

	h = 1.0f/GridSize;

	glBegin(GL_QUADS);
	{
		for ( i=0 ; i<=GridSize ; i++ )
		{
			x = (i-0.5f)*h;
			for ( j=0 ; j<=GridSize ; j++ )
			{
				y = (j-0.5f)*h;
				d00 = dens[Index(i,j)];
				d01 = dens[Index(i,j+1)];
				d10 = dens[Index(i+1,j)];
				d11 = dens[Index(i+1,j+1)];

				glColor3f(d00, d00, d00); glVertex2f(x, y);
				glColor3f(d10, d10, d10); glVertex2f(x+h, y);
				glColor3f(d11, d11, d11); glVertex2f(x+h, y+h);
				glColor3f(d01, d01, d01); glVertex2f(x, y+h);
			}
		}
	}
	glEnd();
}

#endif