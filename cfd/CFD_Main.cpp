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
* <First>       Oct 6, 2013
* <Last>		Oct 6, 2013
* <File>        CFD_Main.cpp
*/

#include <stdlib.h>
#include <stdio.h>
#include <GL/glut.h>
#include <SGE\SGUtils.h>

#include "CFD_Params.h"
#include "resource.h"

using namespace sge;

MainActivity *activity;

/* macros */

#define IX(i,j) ((i)+(GridSize+2)*(j))

/* external definitions (from solver.c) */

extern void dens_step ( int GridSize, float * x, float * x0, float * u, float * v, float diff, float dt );
extern void vel_step ( int GridSize, float * u, float * v, float * u0, float * v0, float visc, float dt );


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


void free_data ( void )
{
	if ( u ) free ( u );
	if ( v ) free ( v );
	if ( u_prev ) free ( u_prev );
	if ( v_prev ) free ( v_prev );
	if ( dens ) free ( dens );
	if ( dens_prev ) free ( dens_prev );
}

void clear_data ( void )
{
	int i, size=(GridSize+2)*(GridSize+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
	}
}

int allocate_data ( void )
{
	int size = (GridSize+2)*(GridSize+2);

	u			= (float *) malloc ( size*sizeof(float) );
	v			= (float *) malloc ( size*sizeof(float) );
	u_prev		= (float *) malloc ( size*sizeof(float) );
	v_prev		= (float *) malloc ( size*sizeof(float) );
	dens		= (float *) malloc ( size*sizeof(float) );	
	dens_prev	= (float *) malloc ( size*sizeof(float) );

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

void draw_velocity ( void )
{
	int i, j;
	float x, y, h;

	h = 1.0f/GridSize;

	glColor3f ( 0.0f, 0.0f, 1.0f );
	glLineWidth ( 1.0f );

	glBegin ( GL_LINES );

		for ( i=1 ; i<=GridSize ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=1 ; j<=GridSize ; j++ ) {
				y = (j-0.5f)*h;

				glVertex2f ( x, y );
				glVertex2f ( x+u[IX(i,j)], y+v[IX(i,j)] );
			}
		}

	glEnd ();
}

void draw_density ( void )
{
	int i, j;
	float x, y, h, d00, d01, d10, d11;

	h = 1.0f/GridSize;

	glBegin ( GL_QUADS );
	{
		for ( i=0 ; i<=GridSize ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=0 ; j<=GridSize ; j++ ) {
				y = (j-0.5f)*h;

				d00 = dens[IX(i,j)];
				d01 = dens[IX(i,j+1)];
				d10 = dens[IX(i+1,j)];
				d11 = dens[IX(i+1,j+1)];

				glColor3f ( d00, d00, d00 ); glVertex2f ( x, y );
				glColor3f ( d10, d10, d10 ); glVertex2f ( x+h, y );
				glColor3f ( d11, d11, d11 ); glVertex2f ( x+h, y+h );
				glColor3f ( d01, d01, d01 ); glVertex2f ( x, y+h );
			}
		}
	}
	glEnd ();
}


/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

void get_from_UI ( float * d, float * u, float * v )
{
#define MouseLeftDown  mouse_down[0]
#define MouseRightDown mouse_down[1]

	int i, j, size = (GridSize+2)*(GridSize+2);

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = d[i] = 0.0f;
	}

	if ( !MouseLeftDown && !MouseRightDown ) return;

	i = (int)((       mx /(float)win_x)*GridSize+1);
	j = (int)(((win_y-my)/(float)win_y)*GridSize+1);

	if ( i<1 || i>GridSize || j<1 || j>GridSize ) return;

	if ( MouseLeftDown ) {
		u[IX(i,j)] = force * (mx-omx);
		v[IX(i,j)] = force * (omy-my);
	}

	if ( MouseRightDown ) {
		d[IX(i,j)] = source;
	}

	omx = mx;
	omy = my;

#undef MouseLeftDown
#undef MouseRightDown
}


/*
  ----------------------------------------------------------------------
   Mainroutine
  ----------------------------------------------------------------------
*/

void key_func ( SG_KEYS key, SG_KEY_STATUS status )
{
	if (status == SG_KEY_STATUS::SG_KEY_DOWN)
	{
		switch ( key )
		{
		case SG_KEYS::SG_KEY_C:
			clear_data ();
			break;
		
		case SG_KEYS::SG_KEY_Q:
			free_data ();
			exit ( 0 );
			break;

		case SG_KEYS::SG_KEY_ESCAPE:
			key_func(SG_KEY_Q, SG_KEY_DOWN);
			break;
		}
	}
}


void mouse_func ( SG_MOUSE mouse, unsigned x, unsigned y )
{
#define MouseLeftDown  mouse_down[0]
#define MouseRightDown mouse_down[1]

	omx = mx = x;
	omx = my = y;

	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN) MouseLeftDown  = true;
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN) MouseRightDown = true;
	if (mouse == SG_MOUSE::SG_MOUSE_MOVE)
	{
		mx = x;
		my = y;
	}

	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP) MouseLeftDown   = false;
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP) MouseRightDown  = false;

#undef MouseLeftDown
#undef MouseRightDown
}


void reshape_func ( unsigned width, unsigned height )
{
	if (height == 0) height = 1;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );

	win_x = width;
	win_y = height;
}


void display_func ( void )
{
	glClear ( GL_COLOR_BUFFER_BIT );
	draw_density();
	draw_velocity();
}


void idle_func( void )
{
	get_from_UI ( dens_prev, u_prev, v_prev );
	vel_step ( GridSize, u, v, u_prev, v_prev, visc, dt );
	dens_step ( GridSize, dens, dens_prev, u, v, diff, dt );
}


int main( int argc, char ** argv )
{
	// Initialize the CFD parameters
	CFD_Init_Params();

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	// Create a main activity and set the window from size as 512x512
	activity = new MainActivity(512, 512);

	// Set application title
	activity->SetApplicationTitle( L"CFD Demo     Version 1.00.01a" );
	activity->SetApplicationIcons(APP_ICONS, APP_ICONS);
		
	// Register callback functions
	activity->RegisterKeyboardFunc ( key_func );
	activity->RegisterMouseFunc ( mouse_func );
	activity->RegisterReshapeFunc ( reshape_func );
	activity->RegisterDisplayFunc ( display_func );
	activity->RegisterIdleFunc ( idle_func );

	// Initialize window
	activity->SetupRoutine();

	// Display and run demo
	activity->MainLoop();

	exit ( 0 );
};