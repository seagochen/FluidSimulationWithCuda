/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
* <File Name>     BasicFunction.cu
*/

#include <time.h>
#include <iostream>
#include <utility>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "Kernels.h"

using namespace sge;

void FluidSimProc::ClearBuffers( void )
{
	DeviceParamDim();

	_zero( gd_density );
	_zero( gd_velocity_u );
	_zero( gd_velocity_v );
	_zero( gd_velocity_w );

	for ( int i = 0; i < m_vectCompBufs.size(); i++ ) _zero( m_vectCompBufs[i] );
		 	 

	for ( int i = 0; i < NODES_X * NODES_Y * NODES_Z; i++ )
	{
		_zero( m_vectGPUDens[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelU[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelV[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUVelW[i] ); _zero( m_vectNewDens[i] );
		_zero( m_vectGPUObst[i] ); _zero( m_vectNewDens[i] );
	}

	if ( helper.GetCUDALastError( "host function failed: ZeroBuffers", __FILE__, __LINE__ ) )
	{
		FreeResource();
		exit( 1 );
	}
};