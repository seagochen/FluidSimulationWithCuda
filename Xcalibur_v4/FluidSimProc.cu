/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 23, 2014
* <File Name>     FluidSimProc.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "FrameworkDynamic.h"
#include "Kernels.h"
#include "SimulationParam.h"

using std::cout;
using std::endl;

using namespace sge;

#define __device_func__ <<<gridDim, blockDim>>>

void FluidSimProc::ClearCompNodes( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectGPUDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelU[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUVelW[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectGPUObst[i], GRIDS_X, GRIDS_Y, GRIDS_Z );

		kernelZeroBuffers __device_func__ ( m_vectNewDens[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelU[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelV[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
		kernelZeroBuffers __device_func__ ( m_vectNewVelW[i], GRIDS_X, GRIDS_Y, GRIDS_Z );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearCompNodes failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	for ( int i = 0; i < m_nNodeNum; i++ )
	{
		cudaMemcpy( m_vectHostDens[i], m_vectNewDens[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelU[i], m_vectNewVelU[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelV[i], m_vectNewVelV[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostVelW[i], m_vectNewVelW[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
		cudaMemcpy( m_vectHostObst[i], m_vectGPUObst[i], sizeof(double) * m_nNodeSize, cudaMemcpyDeviceToHost );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearCompNodes failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::ClearDeviceBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 22, 22, BULLET_X, BULLET_Y, BULLET_Z );

	for ( int i = 0; i < m_vectGPUBuffers.size(); i++ )
	{
		kernelZeroBuffers __device_func__ ( m_vectGPUBuffers[i], BULLET_X, BULLET_Y, BULLET_Z );
	}

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TPBUFFER_S );

	kernelZeroBuffers __device_func__ ( m_ptrGPUShare, TPBUFFER_S );

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}

	cudaMemcpy( m_ptrHostShare, m_ptrGPUShare, sizeof(double) * TPBUFFER_S, cudaMemcpyDeviceToHost );

	if ( m_scHelper.GetCUDALastError( "call member function ClearDeviceBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::ClearVisualBuffers( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, VOLUME_X, VOLUME_Y, VOLUME_Z );

	kernelZeroBuffers __device_func__ ( m_ptrGPUVisual, VOLUME_X, VOLUME_Y, VOLUME_Z );
	cudaMemcpy( m_ptrHostVisual, m_ptrGPUVisual, sizeof(uchar) * m_nVolumSize, cudaMemcpyDeviceToHost );

	if ( m_scHelper.GetCUDALastError( "call member function ClearVisualBuffers failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::PushCompNode( int id )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelLoadBullet __device_func__ ( dev_den, m_vectGPUDens[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_u, m_vectGPUVelU[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_v, m_vectGPUVelV[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_w, m_vectGPUVelW[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );
	kernelLoadBullet __device_func__ ( dev_obs, m_vectGPUObst[id], BULLET_X, BULLET_Y, BULLET_Z, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( m_scHelper.GetCUDALastError( "call member function PushCompNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::PopCompNode( int id )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelExitBullet __device_func__ ( m_vectNewDens[id], dev_den, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelU[id], dev_u, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelV[id], dev_v, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );
	kernelExitBullet __device_func__ ( m_vectNewVelW[id], dev_w, GRIDS_X, GRIDS_Y, GRIDS_Z, BULLET_X, BULLET_Y, BULLET_Z );

	if ( m_scHelper.GetCUDALastError( "call member function PopCompNode failed", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};

void FluidSimProc::GenVolumeImage( FLUIDSPARAM *fluid )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k = 0; k < NODES_Z; k++ )
	{
		for ( int j = 0; j < NODES_Y; j++ )
		{
			for ( int i = 0; i < NODES_X; i++ )
			{
				kernelDensToVolume __device_func__
					( m_ptrGPUVisual, m_vectGPUDens[IX(i,j,k,NODES_X,NODES_Y,NODES_Z)], i, j, k,
					GRIDS_X, GRIDS_Y, GRIDS_Z, VOLUME_X, VOLUME_Y, VOLUME_Z );
			}
		}
	}

	if ( cudaMemcpy( m_ptrHostVisual, m_ptrGPUVisual, sizeof(uchar) * m_nVolumSize, 
		cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "call member function GenVolumeImage failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	
	fluid->volume.ptrData = m_ptrHostVisual;
};

void FluidSimProc::VelocitySolver( cdouble timestep )
{
	// diffuse the velocity field (per axis):
	Diffusion( dev_u0, dev_u, VISOCITY );
	Diffusion( dev_v0, dev_v, VISOCITY );
	Diffusion( dev_w0, dev_w, VISOCITY );
	
	if ( m_scHelper.GetCUDALastError( "host function failed: Diffusion", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );

	if ( m_scHelper.GetCUDALastError( "host function failed: Projection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
	
	// advect the velocity field (per axis):
	Advection( dev_u0, dev_u, timestep, dev_u, dev_v, dev_w );
	Advection( dev_v0, dev_v, timestep, dev_u, dev_v, dev_w );
	Advection( dev_w0, dev_w, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: Advection", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}

	std::swap( dev_u0, dev_u );
	std::swap( dev_v0, dev_v );
	std::swap( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver( cdouble timestep )
{
	Diffusion( dev_den0, dev_den, DIFFUSION );
	std::swap( dev_den0, dev_den );
	Advection ( dev_den, dev_den0, timestep, dev_u, dev_v, dev_w );

	if ( m_scHelper.GetCUDALastError( "host function failed: DensitySolver", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};

void FluidSimProc::SourceSolver( void )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );	

	kernelAddSource __device_func__ ( dev_den, dev_u, dev_v, dev_w, dev_obs, DENSITY, VELOCITY, DELTATIME );
//	if ( decrease_times eqt 0 )
//	{
//		
//
//		if ( helper.GetCUDALastError( "device kernel: kernelPickData failed", __FILE__, __LINE__ ) )
//		{
//			FreeResource();
//			exit( 1 );
//		}
//
//		increase_times++;
//
//		if ( increase_times eqt 200 )
//		{
//			decrease_times = increase_times;
//			increase_times = 0;
//		}
//	}
//	else
//	{
//		decrease_times--;
//	}
};

void FluidSimProc::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	for ( int k=0; k<20; k++)
	{
		kernelJacobi<<<gridDim,blockDim>>>( out, in, diff, divisor);
	}
};

void FluidSimProc::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelAdvection<<<gridDim,blockDim>>>( out, in, timestep, u, v, w );
};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_Y * GRIDS_Z;
	Jacobi ( out, in, rate, 1+6*rate );
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	// the velocity gradient
	kernelGradient<<<gridDim,blockDim>>>( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi(p, div, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract<<<gridDim,blockDim>>>( u, v, w, p );
};