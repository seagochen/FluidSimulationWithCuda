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
* <Last>		Oct 10, 2013
* <File>        Main.cpp
*/

#define _MAIN_CPP_

#include <GL\glew.h>
#include <GL\glut.h>
#include <SGE\SGUtils.h>
#include "macro_def.h"
#include "visual_framework.h"
#include "resource.h"

#define APP_ICONS IDI_ICON1

////////////////////////////////////////////////////////////////////////
///

using namespace sge;

MainActivity  *activity;
Visualization *visual;

///
////////////////////////////////////////////////////////////////////////
///

void key_func(SG_KEYS key, SG_KEY_STATUS status);

void close_func(void);

void mouse_func(SG_MOUSE mouse, unsigned x, unsigned y);

void reshape_func(unsigned width, unsigned height);

void display_func(void);

void idle_func(void);

void param_init(void);

void clear_data(void);

int allocate_data(void);

void cuda_init(void);

///
////////////////////////////////////////////////////////////////////////
///

int main(int argc, char ** argv)
{
	// Create a main activity and set the window from size as 512x512
	activity = new MainActivity(WINDOWSX, WINDOWSY);
	visual   = new Visualization(WINDOWSX, WINDOWSX, activity);

	// Initialize the parameters
	param_init();

	// Initialize the CUDA
	cuda_init();

	// Register callback function to visualization
	visual->RegisterDisplay (display_func);
	visual->RegisterIdle    (idle_func);
	visual->RegisterKeyboard(key_func);
	visual->RegisterMouse   (mouse_func);
	visual->RegisterResize  (reshape_func);
	visual->RegisterDestroy (close_func);

	if ( !allocate_data() ) exit(1);
	clear_data();

	// Set application title
#if GPU_ON
	activity->SetApplicationTitle( L"CFD - Navigator No. I  CUDA version" );
#else
	activity->SetApplicationTitle( L"CFD - Navigator No. I  CPU version" );
#endif
	activity->SetApplicationIcons(APP_ICONS, APP_ICONS);
			
	// Register callback functions
	activity->RegisterCreateFunc   (visual->OnCreate);
	activity->RegisterDestoryFunc  (visual->OnDestroy);
	activity->RegisterKeyboardFunc (visual->OnKeyboard);
	activity->RegisterMouseFunc    (visual->OnMouse);
	activity->RegisterReshapeFunc  (visual->OnResize);
	activity->RegisterDisplayFunc  (visual->OnDisplay);
	activity->RegisterIdleFunc     (visual->OnIdle);

	// Initialize window
	activity->SetupRoutine();

	// Display and run demo
	activity->MainLoop();

	return 0;
};

///
///////////////////////////////////////////////////////////////////////
///

void free_dev_list()
{
	// Release CUDA resource if failed
	for (int i=0; i<devices; i++)
	{
		cudaFree(dev_list[i]);
	}
	dev_list.empty();
}

void cuda_init()
{
#if GPU_ON
	
	// Push dev into vector
	for (int i=0; i<devices; i++)
	{
		static float *ptr;
		dev_list.push_back(ptr);
	}

	size_t size = ENTIRE_GRIDS_NUMBER * ENTIRE_GRIDS_NUMBER;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        free_dev_list();
    }

    // Allocate GPU buffers for three vectors (two input, one output).
	for (int i=0; i<devices; i++)
	{
		cudaStatus = cudaMalloc((void**)&dev_list[i], size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			free_dev_list();
		}
	}

#endif
};

void param_init()
{
	GridSize = GRIDS_WITHOUT_GHOST;
	dt       = DELTA_TIME;
	diff     = DIFFUSION;
	visc     = VISCOSITY;
	force    = FORCE;
	source   = SOURCE;
	win_x    = WINDOWSX;
	win_y    = WINDOWSY;
};

void free_data(void)
{
	if ( u ) SAFE_FREE_PTR(u);
	if ( v ) SAFE_FREE_PTR(v);
	if ( u_prev ) SAFE_FREE_PTR(u_prev);
	if ( v_prev ) SAFE_FREE_PTR(v_prev);
	if ( dens ) SAFE_FREE_PTR(dens);
	if ( dens_prev ) SAFE_FREE_PTR(dens_prev);

#if GPU_ON
	// Release CUDA resources
	for (int i=0; i<devices; i++)
		cudaFree(dev_list[i]);
#endif
}

void clear_data(void)
{
	int size=(GridSize+2)*(GridSize+2);

	for ( int i=0; i<size ; i++ )
	{
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
	}
}

int allocate_data(void)
{
	int size = (GridSize+2)*(GridSize+2);

	u			= (float *)malloc(size*sizeof(float));
	v			= (float *)malloc(size*sizeof(float));
	u_prev		= (float *)malloc(size*sizeof(float));
	v_prev		= (float *)malloc(size*sizeof(float));
	dens		= (float *)malloc(size*sizeof(float));	
	dens_prev	= (float *)malloc(size*sizeof(float));

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) 
	{
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return True;
}

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

void get_from_UI(float * d, float * u, float * v)
{
#define MouseLeftDown  mouse_down[0]
#define MouseRightDown mouse_down[1]

	int i, j, size = ENTIRE_GRIDS_NUMBER * ENTIRE_GRIDS_NUMBER;

	for (i=0 ; i<size ; i++) 
	{
		u[i] = v[i] = d[i] = 0.0f;
	}

	if (!MouseLeftDown && !MouseRightDown) return;

	i = (int)((       mx /(float)win_x)*GridSize+1);
	j = (int)(((win_y-my)/(float)win_y)*GridSize+1);

	if (i<1 || i>GridSize || j<1 || j>GridSize) return;

	if (MouseLeftDown)
	{
		u[Index(i,j)] = force * (mx-omx);
		v[Index(i,j)] = force * (omy-my);
	}

	if (MouseRightDown)
	{
		d[Index(i,j)] = source;
	}

	omx = mx;
	omy = my;

#undef MouseLeftDown
#undef MouseRightDown
}

void key_func(SG_KEYS key, SG_KEY_STATUS status)
{
	if (status == SG_KEY_STATUS::SG_KEY_DOWN)
	{
		switch (key)
		{
		case SG_KEYS::SG_KEY_C:
			clear_data();
			break;
		
		case SG_KEYS::SG_KEY_Q:
			free_data();
			exit ( 0 );
			break;

		case SG_KEYS::SG_KEY_ESCAPE:
			key_func(SG_KEY_Q, SG_KEY_DOWN);
			break;
		}
	}
}

void close_func(void)
{
	free_data();

#if GPU_ON

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }

#endif

	exit(0);
};

void mouse_func(SG_MOUSE mouse, unsigned x, unsigned y)
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

void reshape_func(unsigned width, unsigned height)
{
	if (height == 0) height = 1;

	glViewport(0, 0, width, height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);
	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f);

	win_x = width;
	win_y = height;
}

void display_func(void)
{
	glClear(GL_COLOR_BUFFER_BIT);
	draw_density();
	draw_velocity();
}

void idle_func(void)
{
	get_from_UI(dens_prev, u_prev, v_prev);
	vel_step(u, v, u_prev, v_prev);
	dens_step(dens, dens_prev, u, v);
}

///
///////////////////////////////////////////////////////////////////////