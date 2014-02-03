/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 08, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     BasicKernels.cu
*/

#ifndef __basic_kernels_cu_
#define __basic_kernels_cu_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "DataStructures.h"
#include "CUDAMacroDef.h"


using namespace sge;




__global__ void kernelSmoothHalo
	( double *buffer, double *stores, SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	GetIndex();

	buffer[ Index(i,j,gst_header) ] += atomicTrilinear( stores, global, type, i, j, gst_header );
	buffer[ Index(i,j,gst_tailer) ] += atomicTrilinear( stores, global, type, i, j, gst_tailer );
	buffer[ Index(i,gst_header,k) ] += atomicTrilinear( stores, global, type, i, gst_header, k );
	buffer[ Index(i,gst_tailer,k) ] += atomicTrilinear( stores, global, type, i, gst_tailer, k );
	buffer[ Index(gst_header,j,k) ] += atomicTrilinear( stores, global, type, gst_header, j, k );
	buffer[ Index(gst_tailer,j,k) ] += atomicTrilinear( stores, global, type, gst_tailer, j, k );

	if ( i eqt gst_header or i eqt gst_tailer ) 
	{
		if ( j eqt gst_header or j eqt gst_tailer )
		{
			if ( k eqt gst_header or k eqt gst_tailer )
			{
				/* vertices */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 3.f;
			}
			elif ( k not_eq gst_header and k not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}

	if ( i eqt gst_header or i eqt gst_tailer ) 
	{
		if ( k eqt gst_header or k eqt gst_tailer )
		{
			if ( j not_eq gst_header and j not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}

	if ( j eqt gst_header or j eqt gst_tailer )
	{
		if ( k eqt gst_header or k eqt gst_tailer )
		{
			if ( i not_eq gst_header and i not_eq gst_tailer )
			{
				/* edges x 4 */
				buffer[Index(i,j,k)] = buffer[Index(i,j,k)] / 2.f;
			}
		}
	}
};

__host__ void hostBoundary
	( double *buffer, double *stores, SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	
	kernelBoundary<<<gridDim, blockDim>>>( buffer, global->ptrCenter, type );
	kernelSmoothHalo<<<gridDim, blockDim>>>( buffer, stores, global, type );
};

__global__ void kernelJacobi( double *grid_out, double const *grid_in, 
							 double const diffusion, double const divisor )
{
	GetIndex();
	BeginSimArea();

	double div = 0.f;
	if ( divisor <= 0.f ) div = 1.f;
	else div = divisor;

	grid_out [ Index(i,j,k) ] = 
		( grid_in [ Index(i,j,k) ] + diffusion * 
			(
				grid_out [ Index(i-1, j, k) ] + grid_out [ Index(i+1, j, k) ] +
				grid_out [ Index(i, j-1, k) ] + grid_out [ Index(i, j+1, k) ] +
				grid_out [ Index(i, j, k-1) ] + grid_out [ Index(i, j, k+1) ]
			) 
		) / div;

	EndSimArea();
}

__host__ void hostJacobi
	( double *grid_out, double const *grid_in, double const diffusion )
{
	double rate = diffusion;

	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
		kernelJacobi <<<gridDim, blockDim>>> (grid_out, grid_in, rate, 1+6*rate);
};

__global__ void kernelAdvection
	( double *buffer, double *stores, const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	GetIndex();
	
	double u = i - global->ptrCenter[ Index(i,j,k) ].u * DELTATIME;
	double v = j - global->ptrCenter[ Index(i,j,k) ].v * DELTATIME;
	double w = k - global->ptrCenter[ Index(i,j,k) ].w * DELTATIME;
	buffer[ Index(i,j,k) ] = atomicTrilinear( stores, global, type, u, v, w );
};

__host__ void hostAdvection( double *buffer, double *stores,
								const SGDEVICEBUFF *global, SGFIELDTYPE type )
{
	cudaDeviceDim3D();
	kernelAdvection <<<gridDim, blockDim>>> ( buffer, stores, global, type );
};

__global__ void kernelGradient( double *div, double *p, 
	double const *vel_u, double const *vel_v, double const *vel_w )
{
	GetIndex();
	BeginSimArea();
	
	const double h = 1.f / GRIDS_X;

	// previous instantaneous magnitude of velocity gradient 
	//		= (sum of velocity gradients per axis)/2N:
	div [ Index(i,j,k) ] = -0.5f * h * (
			vel_u [ Index(i+1, j, k) ] - vel_u [ Index(i-1, j, k) ] + // gradient of u
			vel_v [ Index(i, j+1, k) ] - vel_v [ Index(i, j-1, k) ] + // gradient of v
			vel_w [ Index(i, j, k+1) ] - vel_w [ Index(i, j, k-1) ]   // gradient of w
		);
	// zero out the present velocity gradient
	p [ Index(i,j,k) ] = 0.f;
	
	EndSimArea();
};

__global__ void kernelSubtract( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__ void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient <<<gridDim, blockDim>>> (div, p, vel_u, vel_v, vel_w);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 1.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract <<<gridDim, blockDim>>> (vel_u, vel_v, vel_w, p);
};

__global__ void kernelOutFlow( double *buffer, const SGDEVICEBUFF *global, SGFIELDTYPE type, SGNODECOORD coord )
{
	GetIndex();

	switch (coord)
	{
	case sge::SG_LEFT:
		buffer[Index(gst_tailer, j, k)] += atomicGetDeviceBuffer( global, type, gst_header, j, k );
		buffer[Index(gst_tailer, j, k)] /= 2.f;
		break;
	case sge::SG_RIGHT:
		buffer[Index(gst_header, j, k)] += atomicGetDeviceBuffer( global, type, gst_tailer, j, k );
		buffer[Index(gst_header, j, k)] /= 2.f;
		break;
	case sge::SG_UP:
		buffer[Index(i, gst_header, k)] += atomicGetDeviceBuffer( global, type, i, gst_tailer, k );
		buffer[Index(i, gst_header, k)] /= 2.f;
		break;
	case sge::SG_DOWN:
		buffer[Index(i, gst_tailer, k)] += atomicGetDeviceBuffer( global, type, i, gst_header, k );
		buffer[Index(i, gst_tailer, k)] /= 2.f;
		break;
	case sge::SG_FRONT:
		buffer[Index(i, j, gst_header)] += atomicGetDeviceBuffer( global, type, i, j, gst_tailer );
		buffer[Index(i, j, gst_header)] /= 2.f;
		break;
	case sge::SG_BACK:
		buffer[Index(i, j, gst_tailer)] += atomicGetDeviceBuffer( global, type, i, j, gst_header );
		buffer[Index(i, j, gst_tailer)] /= 2.f;
		break;

	default:
		break;
	}
};



__host__ void VelocitySolver
	( double *u, double *v, double *w, double *div, double *p, 
	double *u0, double *v0, double *w0,
	SGDEVICEBUFF *global, double *stores )
{
	cudaDeviceDim3D();
	
	/* copy data to temporary buffer */
	m_helper.CopyData( u, global, SG_VELOCITY_U_FIELD, SG_CENTER );
	m_helper.CopyData( v, global, SG_VELOCITY_V_FIELD, SG_CENTER );
	m_helper.CopyData( w, global, SG_VELOCITY_W_FIELD, SG_CENTER );
	
	/* diffuse the velocity field */
	hostJacobi( u0, u, VISOCITY );
	hostJacobi( v0, v, VISOCITY );
	hostJacobi( w0, w, VISOCITY );
		
		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject( u, v, w, div, p );

		/* boundary condition */
		hostBoundary( u, stores, global, SG_VELOCITY_U_FIELD );
		hostBoundary( v, stores, global, SG_VELOCITY_V_FIELD );
		hostBoundary( w, stores, global, SG_VELOCITY_W_FIELD );
		
		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
		
		/* advect the velocity field (per axis): */
		hostAdvection( u0, stores, global, SG_VELOCITY_U_FIELD );
		hostAdvection( v0, stores, global, SG_VELOCITY_V_FIELD );
		hostAdvection( w0, stores, global, SG_VELOCITY_W_FIELD );

		/* stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field) */
		hostProject ( u0, v0, w0, div, p );

		kernelSwapBuffer<<<gridDim, blockDim>>> ( u0, u );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( v0, v );
		kernelSwapBuffer<<<gridDim, blockDim>>> ( w0, w );

		/* boundary condition */
		hostBoundary( u, stores, global, SG_VELOCITY_U_FIELD );
		hostBoundary( v, stores, global, SG_VELOCITY_V_FIELD );
		hostBoundary( w, stores, global, SG_VELOCITY_W_FIELD );

		/* retrieve data */
		m_helper.CopyData( global, u, SG_VELOCITY_U_FIELD, SG_CENTER );
		m_helper.CopyData( global, v, SG_VELOCITY_V_FIELD, SG_CENTER );
		m_helper.CopyData( global, w, SG_VELOCITY_W_FIELD, SG_CENTER );
	}

	__host__ void DensitySolver( double *dens, double *dens0, SGDEVICEBUFF *global, double *stores )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );
	
		/* copy data to temporary buffer */
		m_helper.CopyData( dens, global, SG_DENSITY_FIELD, SG_CENTER );
	
		/* advection */
		hostJacobi( dens0, dens, DIFFUSION );
		kernelSwapBuffer <<<gridDim, blockDim>>> ( dens0, dens );
		hostAdvection( dens, stores, global, SG_DENSITY_FIELD );

		/* boundary condition */
		hostBoundary( dens, stores, global, SG_DENSITY_FIELD );

		/* retrive data */
		m_helper.CopyData( global, dens, SG_DENSITY_FIELD, SG_CENTER );
	};




	__host__ void HaloDataExchange( double *buffer, SGDEVICEBUFF *global, SGFIELDTYPE type, SGNODECOORD coord )
	{
		dim3 gridDim, blockDim;
		m_helper.DeviceDim3Dx( &gridDim, &blockDim );

		m_helper.CopyData( buffer, global, type, coord );
		kernelOutFlow<<<gridDim, blockDim>>>( buffer, global, type, coord );
		m_helper.CopyData( global, buffer, type, coord );
	};

#endif