/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 19, 2014
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

using namespace sge;

#define __device_func__ <<<gridDim, blockDim>>>

/************************************************************************************
** The following implement functions are member of class NavierStokesSolver        **
*************************************************************************************/

void NavierStokesSolver::AddSource( double *ptrDevDens, double *ptrDevU, double *ptrDevV, double *ptrDevW,
			cdouble *ptrDevObst, int *nInTime, int *nDeTime, cdouble deltatime )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	if ( *nDeTime eqt 0 )
	{
		kernelAddSource __device_func__ ( ptrDevDens, ptrDevU, ptrDevV, ptrDevW, 
			ptrDevObst, DENSITY, VELOCITY, deltatime, time(NULL), BULLET_X, BULLET_Y, BULLET_Z );

		*nInTime++;

		if ( *nInTime eqt 200 )
		{
			*nDeTime = *nInTime;
			*nInTime = 0;
		}
	}
	else
	{
		*nDeTime--;
	}
};

void NavierStokesSolver::SolveDensity( double *ptrDevDens, double *ptrDevDens0,
									   cdouble *ptrDevU, cdouble *ptrDevV, cdouble *ptrDevW,
									   cdouble timestep )
{
	Diffusion( ptrDevDens0, ptrDevDens, DIFFUSION );
	std::swap( ptrDevDens0, ptrDevDens );
	Advection ( ptrDevDens0, ptrDevDens, timestep, ptrDevU, ptrDevV, ptrDevW );
	std::swap( ptrDevDens0, ptrDevDens );
};

void NavierStokesSolver::Advection( double *out, cdouble *in, cdouble timestep, cdouble *u, cdouble *v, cdouble *w )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );

	kernelAdvection __device_func__ ( out, in, timestep, u, v, w, BULLET_X, BULLET_Y, BULLET_Z );
};

void NavierStokesSolver::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, TILE_X, TILE_Y, GRIDS_X, GRIDS_Y, GRIDS_Z );
	
	for ( int k = 0; k < 20; k++ )
		kernelJacobi __device_func__ ( out, in, diff, divisor, BULLET_X, BULLET_Y, BULLET_Z );	
};

void NavierStokesSolver::Diffusion( double *out, cdouble *in, cdouble diff )
{
	double rate = diff * GRIDS_X * GRIDS_X * GRIDS_X;

	Jacobi( out, in, rate, 1+6*rate );
};


/************************************************************************************
** The following functions are member of class FluidSimProc                        **
*************************************************************************************/


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