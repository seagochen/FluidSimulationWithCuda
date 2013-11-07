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
* <Last>		Nov 6, 2013
* <File>        cfd_rendering.cpp
*/

#ifndef __cfd_rendering_cpp_
#define __cfd_rendering_cpp_

#include "macroDef.h"

static const size_t size = Grids_X * Grids_X;
static float dens[size], u[size], v[size];


void DensityInterpolate ( void )
{
	for ( int i = 0; i < Grids_X; i++ )
	{
		for ( int j = 0; j < Grids_X; j++ )
		{
			float var = 0.f;
			
			for ( int k = 0; k < Grids_X; k++ )
			{
				var += host_den [ cudaIndex3D (i, j, k, Grids_X) ];
			}

			dens [ cudaIndex2D (i, j, Grids_X)] = var;
		}
	}
};


void VelocityInterpolate ( void )
{

};


void DrawVelocity ( void )
{
	VelocityInterpolate ( );

	int i, j;
	float x, y, h;

	h = 1.0f/SimArea_X;

	glColor3f(0.0f, 0.0f, 1.0f);
	glLineWidth(1.0f);

	glBegin(GL_LINES);
	{
		for ( i=1 ; i<=SimArea_X ; i++ )
		{
			x = (i-0.5f)*h;
			for ( j=1 ; j<=SimArea_X ; j++ )
			{
				y = (j-0.5f)*h;
				glVertex2f(x, y);
				glVertex2f(x+host_u[Index(i,j)], y+host_v[Index(i,j)]);
			}
		}
	}
	glEnd();
}

void DrawDensity(void)
{
	DensityInterpolate ( );

	int i, j;
	float x, y, h, d00, d01, d10, d11;

	h = 1.0f/SimArea_X;

	glBegin(GL_QUADS);
	{
		for ( i=0 ; i<=SimArea_X ; i++ )
		{
			x = (i-0.5f)*h;
			for ( j=0 ; j<=SimArea_X ; j++ )
			{
				y = (j-0.5f)*h;
				d00 = host_den[Index(i,j)];
				d01 = host_den[Index(i,j+1)];
				d10 = host_den[Index(i+1,j)];
				d11 = host_den[Index(i+1,j+1)];

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