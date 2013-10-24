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
* <Last>		Oct 24, 2013
* <File>        launch_main.cpp
*/

#ifndef __launch_main_cpp_
#define __launch_main_cpp_

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
	activity = new MainActivity(Client_X, Client_X);
	visual   = new Visualization(Client_X, Client_X, activity);

	// Initialize the parameters
	param_init();

	// Initialize the CUDA
	cuda_init();

	if ( !allocate_data() ) exit(1);
	clear_data();

	// Set application title
	activity->SetApplicationTitle( L"CFD - Navigator No. I  CUDA version" );
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
	// Push dev into vector
	for (int i=0; i<devices; i++)
	{
		static float *ptr;
		dev_list.push_back(ptr);
	}

	size_t size = Grids_X * Grids_X;

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
};

void param_init()
{
	GridSize = SimArea_X;
	dt       = DELTA_TIME;
	diff     = DIFFUSION;
	visc     = VISCOSITY;
	force    = FORCE;
	source   = SOURCE;
	win_x    = Client_X;
	win_y    = Client_X;
};

void free_data(void)
{
	if ( u ) SAFE_FREE_PTR(u);
	if ( v ) SAFE_FREE_PTR(v);
	if ( u_prev ) SAFE_FREE_PTR(u_prev);
	if ( v_prev ) SAFE_FREE_PTR(v_prev);
	if ( dens ) SAFE_FREE_PTR(dens);
	if ( dens_prev ) SAFE_FREE_PTR(dens_prev);

	// Release CUDA resources
	for (int i=0; i<devices; i++)
		cudaFree(dev_list[i]);

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

	return 1;
}



void get_from_UI(float * d, float * u, float * v)
{
#define MouseLeftDown  mouse_down[0]
#define MouseRightDown mouse_down[1]

	int i, j, size = Grids_X * Grids_X;

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

///
///////////////////////////////////////////////////////////////////////

#endif