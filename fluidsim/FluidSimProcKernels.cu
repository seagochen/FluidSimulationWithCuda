/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Nov 15, 2013
* <Last Time>     Feb 15, 2014
* <File Name>     FluidSimProc.cu
*/

#include <stdlib.h>
#include <stdio.h>

#define DEVLISTNUM       10
#define dev_u     dev_bufs[0]
#define dev_v     dev_bufs[1]
#define dev_w     dev_bufs[2]
#define dev_div   dev_bufs[3]
#define dev_p     dev_bufs[4]
#define dev_dens  dev_bufs[5]
#define dev_u0    dev_bufs[6]
#define dev_v0    dev_bufs[7]
#define dev_w0    dev_bufs[8]
#define dev_dens0 dev_bufs[9]

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "FluidSimProc.h"

using namespace sge;

/* 默认的构造函数，分配流体模拟所需要的空间，以及初始化相关参数 */
FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
{
	/* initialize the parameters of fluid simulation */
	InitParams( fluid );

	/* allocate the space for fluid simulation */
	if ( !AllocateResource() )
	{
		FreeResource();
		printf(" malloc buffers for fluid simulation failed! \n");
		exit(1);
	}

	/* initialize the nodes */
	InitSimNodes();

	/* building structure order */
	BuildOrder();

	/* select and active a node for fluid simulation */
	ActiveNode( 1, 1, 0 );
	
	/* finally, print the state message and zero the data */
	printf( "fluid simulation ready...\n" );
};

/* 初始化流体模拟的相关参数 */
void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	/* initilize the status of FPS counter */
	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;
};

/* fluid simulation processing function */
void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( fluid->run )
	{
	}
};

/* allocate resource */
bool FluidSimProc::AllocateResource( void )
{
	return true;
};

/* when program existed, release resource */
void FluidSimProc::FreeResource( void )
{};

/* zero the buffers for fluid simulation */
void FluidSimProc::ZeroBuffers( void )
{};

/* choose the node and mark it as actived */
void FluidSimProc::ActiveNode( int i, int j, int k )
{};

/* choose the node and mark it as deactived */
void FluidSimProc::DeactiveNode( int i, int j, int k )
{};

/* retrieve the density back and load into volumetric data for rendering */
void FluidSimProc::PickVolumetric( FLUIDSPARAM *fluid )
{};

/* copy host data to CUDA device */
void FluidSimProc::UploadBuffers( void )
{};

/* retrieve data back to host */
void FluidSimProc::DownloadBuffers( void )
{};

/* zero data, set the bounds */
void FluidSimProc::InitSimNodes( void )
{};

/* create simulation nodes' topological structure */
void FluidSimProc::BuildOrder( void )
{};