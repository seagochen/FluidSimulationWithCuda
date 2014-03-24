/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Mar 24, 2014
* <File Name>     FluidSimProc.cpp
*/

#include <time.h>
#include <iostream>
#include <utility>
#include <device_launch_parameters.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "Kernels.h"

using namespace sge;
using std::cout;
using std::endl;

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "cannot set device", __FILE__, __LINE__ );
		exit(1);
	}

	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	AllocateResource();

	/* build order */
	CreateTopology();
	
	/* clear buffer */
	ClearBuffers();

	/* set boundary */
	InitBoundary();

	/* finally, print message */
	printf( "fluid simulation ready...\n" );
};

void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS = 0;

	m_node_size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof(double);
	m_volm_size = VOLUME_X * VOLUME_X * VOLUME_X * sizeof(SGUCHAR);

	increase_times = decrease_times = 0;

	ptr = nullptr;
	m_cursor.x = m_cursor.y = m_cursor.z = 0;

	m_szTitle = "Excalibur OTL 4.00.00. FPS: %d ";
};


bool CreateCompNodesForDevice( vector<double*> *vectDens,  
							  vector<double*> *vectVelU,
							  vector<double*> *vectVelV,
							  vector<double*> *vectVelW,
							  vector<double*> *vectObst,
							  FunctionHelper *helper, 
							  size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};

bool CreateCompNodesForDevice( vector<double*> *vectBuf, FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptrD );
	}

	return true;
};

bool CreateCompNodesForHost( vector<double*> *vectDens,  
							  vector<double*> *vectVelU,
							  vector<double*> *vectVelV,
							  vector<double*> *vectVelW,
							  vector<double*> *vectObst,
							  FunctionHelper *helper, 
							  size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateHostBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};

void FluidSimProc::AllocateResource( void )
{
	if ( not CreateCompNodesForHost(
		&m_vectHostDens, 
		&m_vectHostVelU,
		&m_vectHostVelV,
		&m_vectHostVelW, 
		&m_vectHostObst,
		&m_scHelper, 
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes for host failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice(
		&m_vectGPUDens, 
		&m_vectGPUVelU,
		&m_vectGPUVelV,
		&m_vectGPUVelW,
		&m_vectGPUObst,
		&m_scHelper,
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes layer 0 for device failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice(
		&m_vectNewDens, 
		&m_vectNewVelU,
		&m_vectNewVelV,
		&m_vectNewVelW,
		&m_vectNewObst,
		&m_scHelper,
		GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double),
		NODES_X * NODES_Y * NODES_Z ))
	{
		cout << "create computation nodes layer 1 for device failed" << endl;
		FreeResource();
		exit(1);
	}

	if ( not CreateCompNodesForDevice( &m_vectCompBufs, &m_scHelper, GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 40 ) )
	{
		cout << "create computation buffers for device failed" << endl;
		FreeResource();
		exit(1);
	}

	helper.CreateDeviceBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &gd_density );
	helper.CreateDeviceBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &gd_obstacle );
	helper.CreateDeviceBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &gd_velocity_u );
	helper.CreateDeviceBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &gd_velocity_v );
	helper.CreateDeviceBuffers( GRIDS_X * GRIDS_Y * GRIDS_Z * sizeof(double), 1, &gd_velocity_w );
	helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(double), 1, &dev_dtpbuf );
	helper.CreateDeviceBuffers( TPBUFFER_X*sizeof(int), 1, &dev_ntpbuf );
	helper.CreateHostBuffers( TPBUFFER_X*sizeof(double), 1, &host_dtpbuf );
	helper.CreateHostBuffers( TPBUFFER_X*sizeof(int), 1, &host_ntpbuf );
	helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual );
	helper.CreateHostBuffers( m_volm_size, 1, &host_visual );

	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		node->active   = false;
		gpu_node.push_back( node );
	}

	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{		
		SimNode *node  = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft  = node->ptrRight = nullptr;
		node->ptrDown  = node->ptrUp = nullptr;
		node->updated  = false;
		node->active   = false;
		host_node.push_back( node );
	}
};