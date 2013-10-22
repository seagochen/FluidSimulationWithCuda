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
* <Last>		Oct 22, 2013
* <File>        cfd_main.cpp
*/

#ifndef __cfd_main_cpp_
#define __cfd_main_cpp_

#include "stdafx.h"
using namespace sge;

////////////////////////////////////////////////////////////////////////
///

MainActivity  *activity;
Visualization *visual;

cudaError      cudaStatus;

SGRUNTIMEMSG   clear_data( void );
SGRUNTIMEMSG   runtime_init( void );

///
////////////////////////////////////////////////////////////////////////
///

int main(int argc, char ** argv)
{
	// Create a main activity and set the window from size as 512x512
	visual   = new Visualization ( ClientSize_X, ClientSize_X, activity );

	// Initialize the CUDA
	if ( runtime_init() != SG_RUNTIME_OK )
	{
		ErrorMSG("runtime_init failed!");
		exit(1);
	}

	if ( clear_data() != SG_RUNTIME_OK )
	{
		ErrorMSG("clear_data failed!");
		exit(1);
	}

	// Set application title
	activity->SetApplicationTitle  ( L"CFD - Navigator No. I  CPU version" );
	activity->SetApplicationIcons  ( APP_ICON, APP_ICON );
			
	// Register callback functions
	activity->RegisterCreateFunc   ( visual->OnCreate );
	activity->RegisterDestoryFunc  ( visual->OnDestroy );
	activity->RegisterKeyboardFunc ( visual->OnKeyboard );
	activity->RegisterMouseFunc    ( visual->OnMouse );
	activity->RegisterReshapeFunc  ( visual->OnResize );
	activity->RegisterDisplayFunc  ( visual->OnDisplay );
	activity->RegisterIdleFunc     ( visual->OnIdle );

	// Initialize window
	activity->SetupRoutine ();

	// Display and run demo
	activity->MainLoop ();

	return 0;
};


SGRUNTIMEMSG runtime_init ( void )
{
	// Push dev into vector
	for ( int i=0; i<dev_num; i++ )
	{
		static float *ptr;
		dev_list.push_back ( ptr );
	}
	for ( int i=0; i<host_num; i++ )
	{
		static float *ptr;
		host_list.push_back ( ptr );
	}

	size_t size = Grids_X * Grids_X;

    // Choose which GPU to run on, change this on a multi-GPU system
    cudaStatus = cudaSetDevice ( 0 );
    if ( cudaStatus != cudaSuccess ) {
        fprintf( stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" );
		goto Error;
    }

    // Allocate GPU buffers
	for ( int i=0; i < dev_num; i++ )
	{
		cudaStatus = cudaMalloc ( ( void** ) &dev_list[i], size * sizeof ( float ) );
		if ( cudaStatus != cudaSuccess ) {
			fprintf ( stderr, "cudaMalloc failed!" );
			goto Error;
		}
	}
	
	// Allocate host memory
	for ( int i=0; i < host_num; i++ )
	{
		host_list[i] = ( float* ) malloc ( size * sizeof ( float ) );
	}

	return SG_RUNTIME_OK;


Error:
	// Release CUDA resource if failed
	for ( int i=0; i < dev_num; i++ )
	{
		cudaFree ( dev_list[i] );
	}
	dev_list.empty ();

	// Release other resource
	for ( int i=0; i < host_num; i++ )
	{
		SAFE_FREE_PTR ( host_list[i] );
	}
	host_list.empty ();

	return SG_NEW_OBJECT_FAILED_EXCEP;
};


SGRUNTIMEMSG clear_data ( void )
{
	int size = Grids_X * Grids_X;

	if ( host_den == NULL || host_den0 == NULL  || host_v == NULL || host_v0 == NULL  || host_u == NULL || host_u0 == NULL )
	{
		SGRUNTIMEMSG::SG_NULL_POINTER_EXCEP;
	}

	for ( int i = 0; i < size ; i++ )
	{
		host_den[i] = host_den0[i] = 0.f;
		host_u[i] = host_u0[i] = 0.f;
		host_v[i] = host_v0[i] = 0.f;
	}

	return SGRUNTIMEMSG::SG_RUNTIME_OK;

}

///
///////////////////////////////////////////////////////////////////////

#endif