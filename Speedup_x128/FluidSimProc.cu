/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Apr 03, 2014
* <File Name>     FluidSimProc.cu
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
using std::cout;
using std::endl;

static int t_totaltimes = 0;

static clock_t t_start, t_finish;
static double t_duration;

static clock_t t_estart, t_efinish;
static double t_eduration;

FluidSimProc::FluidSimProc( FLUIDSPARAM *fluid )
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
	
	/* clear buffer */
	ClearBuffers();

	/* create boundary condition */
	InitBoundary();

	/* finally, print message */
	printf( "fluid simulation ready!\n" );
};


void FluidSimProc::InitParams( FLUIDSPARAM *fluid )
{
	fluid->fps.dwCurrentTime    = 0;
	fluid->fps.dwElapsedTime    = 0;
	fluid->fps.dwFrames         = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS             = 0;

	srand(time(NULL));

	m_szTitle = APP_TITLE;

	t_estart = clock();
};


void FluidSimProc::AllocateResource( void )
{
	/* 为全局计算节点分配空间，需要创建5个： u, v, w, obstacle, density */
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevGlobalx, 
		GLOBAL_X * GLOBAL_Y * GLOBAL_Z * sizeof(double), STANDARD ) ) goto VectError;
	else
	{
		cout << "buffers for keeping global flux were created" << endl
			<< "size of each is " << GLOBAL_X * GLOBAL_Y * GLOBAL_Z << " sizeof(double)" << endl
			<< "number of vector is " << m_vectDevGlobalx.size() << endl;
	}

	/* 为全局计算缓存分配空间，需要创建11个： u, v, w, obst, dens, u0, v0, w0, dens0, div, p */
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevGlobalBx,
		GLOBAL_BX * GLOBAL_BY * GLOBAL_BZ * sizeof(double), EXTENDED ) ) goto VectError;
	else
	{
		cout << "buffers for global flux computation were created" << endl
			<< "size of each is " << GLOBAL_BX * GLOBAL_BY * GLOBAL_BZ << " sizeof(double)" << endl
			<< "number of vector is " << m_vectDevGlobalBx.size() << endl;
	}

	/* 为扩展缓存分配空间，需要创建5个： u, v, w, obstacle, density */
	if ( not m_scHelper.CreateCompNodesForDevice( &m_vectDevExtend,
		EXTEND_X * EXTEND_Y * EXTEND_Z * sizeof(double), STANDARD ) ) goto VectError;
	else
	{
		cout << "extended buffers for keeping global data were created" << endl
			<< "size of each is " << EXTEND_X * EXTEND_Y * EXTEND_Z << " sizeof(doube)" << endl
			<< "number of vector is " << m_vectDevExtend.size() << endl;
	}

	/* 为小计算节点分配空间，需要创建 5 * 4³ 个 */
	if ( not m_scHelper.CreateCompNodesForDevice(&m_vectDevSubNodex,
		SUBNODE_X * SUBNODE_Y * SUBNODE_Z * sizeof(double), STANDARD * 4 * 4 * 4 ) ) goto VectError;
	else
	{
		cout << "buffers for sub-node were created " << endl
			<< "size of each is " << SUBNODE_X * SUBNODE_Y * SUBNODE_Z << " sizeof(double)" << endl
			<< "number of vector is " << m_vectDevSubNodex.size() << endl;
	}

	/* 为小计算缓存分配空间，需要创建 11 * 4³ */
	if ( not m_scHelper.CreateCompNodesForDevice(&m_vectDevSubNodeBx,
		SUBNODE_BX * SUBNODE_BY * SUBNODE_BZ * sizeof(double), EXTENDED * 4 * 4 * 4 ) ) goto VectError;
	{
		cout << "buffers for sub-node computation were created " << endl
			<< "size of each is " << SUBNODE_BX * SUBNODE_BY * SUBNODE_BZ << " sizeof(double)" << endl
			<< "number of vector is " << m_vectDevSubNodeBx.size() << endl;
	}

	goto NextStep;

VectError:
		cout << "create computation buffers for device failed" << endl;
		FreeResource();
		exit(1);

NextStep:

		/* 创建用于计算 ∑ρ 的缓存*/
		if ( m_scHelper.CreateDeviceBuffers( sizeof(double) * 4 * 4 * 4,
			1, &m_ptrDevSum ) not_eq SG_RUNTIME_OK ) goto BufsError;
		if ( m_scHelper.CreateHostBuffers( sizeof(double) * 4 * 4 * 4, 
			1, &m_ptrHostSum ) not_eq SG_RUNTIME_OK ) goto BufsError;

		/* 创建体渲染所需的数据 */
		if ( m_scHelper.CreateDeviceBuffers( sizeof(uchar) * VOLUME_X * VOLUME_Y * VOLUME_Z,
			1, &m_ptrDevVisual ) not_eq SG_RUNTIME_OK ) goto BufsError;
		if ( m_scHelper.CreateHostBuffers( sizeof(uchar) * VOLUME_X * VOLUME_Y * VOLUME_Z,
			1, &m_ptrHostVisual ) not_eq SG_RUNTIME_OK ) goto BufsError;

		goto Success;

BufsError:
		cout << "create buffers for device or host failed" << endl;
		FreeResource();
		exit(1);

Success:

		cout << "all resource created" << endl;
};


void FluidSimProc::FreeResource( void )
{
	int i;

	/* 释放全局节点及计算缓存 */
	for ( i = 0; i < m_vectDevGlobalx.size(); i++ ) 
		m_scHelper.FreeDeviceBuffers( 1, &m_vectDevGlobalx[i] );
	
	for ( i = 0; i < m_vectDevGlobalBx.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectDevGlobalBx[i] );

	/* 释放小节点及计算缓存 */
	for ( i = 0; i < m_vectDevSubNodex.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectDevSubNodex[i] );
	for ( i = 0; i < m_vectDevSubNodeBx.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectDevSubNodeBx[i] );

	/* 释放扩展缓存 */
	for ( i = 0; i < m_vectDevExtend.size(); i++ )
		m_scHelper.FreeDeviceBuffers( 1, &m_vectDevExtend[i] );

	/* 释放其他数据 */
	m_scHelper.FreeDeviceBuffers( 2, &m_ptrDevVisual, &m_ptrDevSum );
	m_scHelper.FreeHostBuffers( 2, &m_ptrHostVisual, &m_ptrHostSum );

	t_efinish = clock();
	t_eduration = (double)( t_efinish - t_estart ) / CLOCKS_PER_SEC;

	printf( "total duration: %f\n", t_eduration );
}


void FluidSimProc::RefreshStatus( FLUIDSPARAM *fluid )
{
	/* counting FPS */
	fluid->fps.dwFrames ++;
	fluid->fps.dwCurrentTime = GetTickCount();
	fluid->fps.dwElapsedTime = fluid->fps.dwCurrentTime - fluid->fps.dwLastUpdateTime;

	/* 1 second */
	if ( fluid->fps.dwElapsedTime >= 1000 )
	{
		fluid->fps.uFPS     = fluid->fps.dwFrames * 1000 / fluid->fps.dwElapsedTime;
		fluid->fps.dwFrames = 0;
		fluid->fps.dwLastUpdateTime = fluid->fps.dwCurrentTime;
	}

	/* updating image */
	if ( cudaMemcpy( m_ptrHostVisual, m_ptrDevVisual, VOLUME_X * VOLUME_Y * VOLUME_Z * sizeof(SGUCHAR),
		cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
	{
		m_scHelper.GetCUDALastError( "call member function RefreshStatus failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
	fluid->volume.ptrData = m_ptrHostVisual;
};


void FluidSimProc::ClearBuffers( void )
{
	/* 设定CUDA调用参数, block(32 x 32 threads), dim(128 x 128 x 128) */
	m_scHelper.DeviceParamDim(&gridDim, &blockDim, THREADS_S, 32, 32, 128, 128, 128 );

	/* 首先清理维度一样的体渲染缓存，及拓展缓存 */
	kernelZeroBuffers __device_func__ ( m_ptrDevVisual, VOLUME_X, VOLUME_Y, VOLUME_Z );
	
	int i;
	for ( i = 0; i < m_vectDevExtend.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectDevExtend[i], EXTEND_X, EXTEND_Y, EXTEND_Z );

	/* 检测这次清理是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function ClearBuffers failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 设定CUDA调用参数, block(32 x 32 threads), dim(64 x 64 x 64) */
	m_scHelper.DeviceParamDim(&gridDim, &blockDim, THREADS_S, 32, 32, 64, 64, 64 );

	/* 然后清理一次全局计算缓存 */
	for ( i = 0; i < m_vectDevGlobalx.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectDevGlobalx[i], GLOBAL_X, GLOBAL_Y, GLOBAL_Z );

	/* 检测这次清理是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function ClearBuffers failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 设定CUDA调用参数, block(22 X 22 threads), dim(66 X 66 X 66) */
	m_scHelper.DeviceParamDim(&gridDim, &blockDim, THREADS_S, 22, 22, 66, 66, 66 );

	for ( i = 0; i < m_vectDevGlobalBx.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectDevGlobalBx[i], GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	/* 检测这次清理是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function ClearBuffers failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 清理节点的计算缓存 */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 34, 17, 34, 34, 34 );

	for ( int i = 0; i < m_vectDevSubNodeBx.size(); i++ )
		kernelZeroBuffers __device_func__ ( m_vectDevSubNodeBx[i], SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ );


	goto Success;


Error:
	FreeResource();
	exit(1);

Success:
	cout << "call member function ClearBuffers success" << endl;
}

void FluidSimProc::InitBoundary( void )
{
	/* 设定CUDA调用参数, block(32 x 32), dim(64 x 64 x 64) */
	m_scHelper.DeviceParamDim(&gridDim, &blockDim, THREADS_S, 32, 32, 64, 64, 64 );

	/* 调用CUDA 函数设置边界条件 */
	kernelSetBound __device_func__ ( m_vectDevGlobalx[DEV_OBSTACLE], GLOBAL_X, GLOBAL_Y, GLOBAL_Z );

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 将初始条件全部压入计算缓存中，然后开始流体计算 */
	kernelLoadBullet __device_func__ ( m_vectDevGlobalBx[DEV_DENSITY], m_vectDevGlobalx[DEV_DENSITY],
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ,
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z );

	kernelLoadBullet __device_func__ ( m_vectDevGlobalBx[DEV_VELOCITY_U], m_vectDevGlobalx[DEV_VELOCITY_U],
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ,
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z );

	kernelLoadBullet __device_func__ ( m_vectDevGlobalBx[DEV_VELOCITY_V], m_vectDevGlobalx[DEV_VELOCITY_V],
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ,
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z );

	kernelLoadBullet __device_func__ ( m_vectDevGlobalBx[DEV_VELOCITY_W], m_vectDevGlobalx[DEV_VELOCITY_W],
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ,
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z );

	kernelLoadBullet __device_func__ ( m_vectDevGlobalBx[DEV_OBSTACLE], m_vectDevGlobalx[DEV_OBSTACLE],
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ,
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z );

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 调整CUDA参数，设置扩展缓存的边界条件，block(32 x 32), dim(128 x 128 x 128) */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 128, 128, 128 );

	kernelUpScalingInterpolation __device_func__ 
		( m_vectDevExtend[DEV_OBSTACLE], m_vectDevGlobalx[DEV_OBSTACLE], // dst, src buffers
		GLOBAL_X, GLOBAL_Y, GLOBAL_Z,  // src dim
		EXTEND_X, EXTEND_Y, EXTEND_Z,  // dst dim
		2, 2, 2 );                     // zoom-up rate

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed",
		__FILE__, __LINE__ ) ) goto Error;

	/* 调整CUDA参数，设置子节点的边界条件, block(34 x 17), dim(34 x 34 x 34) */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 34, 17, 34, 34, 34 );

	int ind;

	for ( int k = 0; k < 4; k++ )
	{
		for ( int j = 0; j < 4; j++ )
		{
			for ( int i = 0; i < 4; i++ ) 
			{
				ind = ix( i, j, k, 4, 4 ) * EXTENDED + DEV_OBSTACLE;

				kernelFillBullet __device_func__ 
					( m_vectDevSubNodeBx[ind], m_vectDevExtend[DEV_OBSTACLE], // dst, src,
					EXTEND_X, EXTEND_Y, EXTEND_Z, // src dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ, // dst dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst grd dim
					i, j, k ); // offset
			}
		}
	}

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function InitBoundary failed",
		__FILE__, __LINE__ ) ) goto Error;

	goto Success;


Error:
	FreeResource();
	exit(1);

Success:
	cout << "call member function InitBoundary success" << endl;
};


void FluidSimProc::GenerateVolumeData( void )
{ 
	/* waiting for all kernels end */
	if ( cudaThreadSynchronize() not_eq cudaSuccess )
	{
		printf( "cudaThreadSynchronize failed\n" );
		FreeResource();
		exit( 1 );
	}

	/* 将修正后的流体信息转换为体渲染数据, 不过首先需要先修正CUDA调用参数 */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 32, 32, 32 );

	/* 从计算缓存中取回密度信息 */
	int indX, indEx;
	for ( int k = 0; k < 4; k++ )
	{
		for ( int j = 0; j < 4; j++ )
		{
			for ( int i = 0; i < 4; i++ )
			{
				indX  = ix( i, j, k, 4, 4 ) * STANDARD + DEV_DENSITY;
				indEx = ix( i, j, k, 4, 4 ) * EXTENDED + DEV_DENSITY;

				kernelExitBullet __device_func__ (
					m_vectDevSubNodex[indX], m_vectDevSubNodeBx[indEx], // dst, src
					SUBNODE_X,  SUBNODE_Y,  SUBNODE_Z, // dst dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ ); // src dim
			}
		}
	}

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function GenerateVolumeData failed",
		__FILE__, __LINE__ ) ) goto Error;


	/* 将子节点密度数据转换为渲染所需的体数据 */
	for ( int k = 0; k < 4; k++ )
	{
		for ( int j = 0; j < 4; j++ )
		{
			for ( int i = 0; i < 4; i++ )
			{
				indX = ix( i, j, k, 4, 4 ) *  STANDARD + DEV_DENSITY;
				kernelPickData __device_func__ ( 
					m_ptrDevVisual, m_vectDevSubNodex[indX], // dst, src
					VOLUME_X,  VOLUME_Y,  VOLUME_Z, // src dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst dim
					i, j, k ); // offset
			}
		}
	}

	/* 检测调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function GenerateVolumeData failed",
		__FILE__, __LINE__ ) ) goto Error;

	goto Success;

Error:
	FreeResource();
	exit(1);

Success:
	;
};

void FluidSimProc::SolveGlobalFlux( void )
{
	/* 链接全局计算缓存 */
	dev_den  = &m_vectDevGlobalBx[DEV_DENSITY];
	dev_den0 = &m_vectDevGlobalBx[DEV_DENSITY0];
	dev_u    = &m_vectDevGlobalBx[DEV_VELOCITY_U];
	dev_v    = &m_vectDevGlobalBx[DEV_VELOCITY_V];
	dev_w    = &m_vectDevGlobalBx[DEV_VELOCITY_W];
	dev_u0   = &m_vectDevGlobalBx[DEV_VELOCITY_U0];
	dev_v0   = &m_vectDevGlobalBx[DEV_VELOCITY_V0];
	dev_w0   = &m_vectDevGlobalBx[DEV_VELOCITY_W0];
	dev_obs  = &m_vectDevGlobalBx[DEV_OBSTACLE];
	dev_p    = &m_vectDevGlobalBx[DEV_PRESSURE];
	dev_div  = &m_vectDevGlobalBx[DEV_DIVERGENCE];

	/* 做一次全局计算 */
	SolveNavierStokesEquation( DELTATIME, true, true, true, 
		32, 32, GLOBAL_X, GLOBAL_Y, GLOBAL_Z, GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	/* 检测计算是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed",
		__FILE__, __LINE__ ) )
	{
		FreeResource();
		exit(1);
	}
};


/* 当第一步计算完毕后，从全局数据中采集数据并写入各节点中 */
void FluidSimProc::InterpolationData( void )
{
	/* 从全局缓存中取出数据，需要调整参数block(32 x 32), dim(64 x 64 x 64) */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 64, 64, 64 );

	/* 数据退出计算缓存 */
	kernelExitBullet __device_func__ ( m_vectDevGlobalx[DEV_DENSITY], m_vectDevGlobalBx[DEV_DENSITY],
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z,
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	kernelExitBullet __device_func__ ( m_vectDevGlobalx[DEV_VELOCITY_U], m_vectDevGlobalBx[DEV_VELOCITY_U],
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z,
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	kernelExitBullet __device_func__ ( m_vectDevGlobalx[DEV_VELOCITY_V], m_vectDevGlobalBx[DEV_VELOCITY_V],
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z,
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	kernelExitBullet __device_func__ ( m_vectDevGlobalx[DEV_VELOCITY_W], m_vectDevGlobalBx[DEV_VELOCITY_W],
		GLOBAL_X,  GLOBAL_Y,  GLOBAL_Z,
		GLOBAL_BX, GLOBAL_BY, GLOBAL_BZ );

	/* 检测计算是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed",
		__FILE__, __LINE__ ) ) goto Error;



	/* 调整CUDA 参数，将全局数据采集至扩展缓存中, block(32 x 32), dim(128 x 128 x 128) */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 32, 32, 128, 128, 128 );

	/* 采集数据至扩展缓存 */
	kernelUpScalingInterpolation __device_func__ 
		( m_vectDevExtend[DEV_DENSITY], m_vectDevGlobalx[DEV_DENSITY], // dst, src buffers
		GLOBAL_X, GLOBAL_Y, GLOBAL_Z,  // src dim
		EXTEND_X, EXTEND_Y, EXTEND_Z,  // dst dim
		2, 2, 2 );                     // zoom-up rate

	kernelUpScalingInterpolation __device_func__ 
		( m_vectDevExtend[DEV_VELOCITY_U], m_vectDevGlobalx[DEV_VELOCITY_U], // dst, src buffers
		GLOBAL_X, GLOBAL_Y, GLOBAL_Z,  // src dim
		EXTEND_X, EXTEND_Y, EXTEND_Z,  // dst dim
		2, 2, 2 );                     // zoom-up rate

	kernelUpScalingInterpolation __device_func__ 
		( m_vectDevExtend[DEV_VELOCITY_V], m_vectDevGlobalx[DEV_VELOCITY_V], // dst, src buffers
		GLOBAL_X, GLOBAL_Y, GLOBAL_Z,  // src dim
		EXTEND_X, EXTEND_Y, EXTEND_Z,  // dst dim
		2, 2, 2 );                     // zoom-up rate

	kernelUpScalingInterpolation __device_func__ 
		( m_vectDevExtend[DEV_VELOCITY_W], m_vectDevGlobalx[DEV_VELOCITY_W], // dst, src buffers
		GLOBAL_X, GLOBAL_Y, GLOBAL_Z,  // src dim
		EXTEND_X, EXTEND_Y, EXTEND_Z,  // dst dim
		2, 2, 2 );                     // zoom-up rate

	/* 检测计算是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed",
		__FILE__, __LINE__ ) ) goto Error;



	/* 调整CUDA参数，将拓展缓存的数据转换为节点数据，并存入子计算缓存。*/
	/* 设置的CUDA参数为，block(34 x 17), dim(34 x 34 x 34) */
	
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 34, 17, 34, 34, 34 );
	
	int indDens, indVelU, indVelV, indVelW;
	for ( int k = 0; k < 4; k++ )
	{
		for ( int j = 0; j < 4; j++ )
		{
			for ( int i = 0; i < 4; i++ )
			{
				indDens = ix(i, j, k, 4, 4) * EXTENDED + DEV_DENSITY;
				indVelU = ix(i, j, k, 4, 4) * EXTENDED + DEV_VELOCITY_U;
				indVelV = ix(i, j, k, 4, 4) * EXTENDED + DEV_VELOCITY_V;
				indVelW = ix(i, j, k, 4, 4) * EXTENDED + DEV_VELOCITY_W;

				kernelFillBullet __device_func__ 
					( m_vectDevSubNodeBx[indDens], m_vectDevExtend[DEV_DENSITY], // dst, src
					EXTEND_X, EXTEND_Y, EXTEND_Z, // src dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ, // dst dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst grd dim
					i, j, k ); // offset

				kernelFillBullet __device_func__ 
					( m_vectDevSubNodeBx[indVelU], m_vectDevExtend[DEV_VELOCITY_U], // dst, src
					EXTEND_X, EXTEND_Y, EXTEND_Z, // src dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ, // dst dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst grd dim
					i, j, k ); // offset

				kernelFillBullet __device_func__ 
					( m_vectDevSubNodeBx[indVelV], m_vectDevExtend[DEV_VELOCITY_V], // dst, src
					EXTEND_X, EXTEND_Y, EXTEND_Z, // src dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ, // dst dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst grd dim
					i, j, k ); // offset

				kernelFillBullet __device_func__ 
					( m_vectDevSubNodeBx[indVelW], m_vectDevExtend[DEV_VELOCITY_W], // dst, src
					EXTEND_X, EXTEND_Y, EXTEND_Z, // src dim
					SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ, // dst dim
					SUBNODE_X, SUBNODE_Y, SUBNODE_Z, // dst grd dim
					i, j, k ); // offset
			}
		}
	}

	/* 检测计算是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveGlobalFlux failed",
		__FILE__, __LINE__ ) ) goto Error;

	goto Success;

Error:

	FreeResource();
	exit(1);

Success:
	;
};


void FluidSimProc::SolveNodeFlux( void )
{
	/* 采集数据，并写入各节点中 */
	InterpolationData();

	/* 清理∑缓存 */
	kernelZeroBuffers <<< 1, 64 >>> ( m_ptrDevSum, 64 );

	/* 计算各子节点缓存的∑值 */
	m_scHelper.DeviceParamDim( &gridDim, &blockDim, THREADS_S, 34, 17, 34, 34, 34 );
	for ( int i = 0; i < 64; i++ )
	{
		kernelSumDensity __device_func__ 
			( m_ptrDevSum, m_vectDevSubNodeBx[i * EXTENDED + DEV_DENSITY], 
			i, SUBNODE_BX, SUBNODE_BY, SUBNODE_BZ );
	}

	cudaMemcpy( m_ptrHostSum, m_ptrDevSum, sizeof(double) * 64, cudaMemcpyDeviceToHost );

	/* 检测上次调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveNodeFlux failed", 
		__FILE__, __LINE__ ) ) goto Error;

	goto Success;

Error:
	FreeResource();
	exit(1);

Success:

#if 0
	/* 测试数据 */
	system("cls");
#endif 

	for ( int i = 0; i < 64; i++ )
	{
		if ( m_ptrHostSum[i] > 5.f )
		{
#if 0
			printf( "%d, density: %f\n", i, m_ptrHostSum[i] );
#endif 

			dev_den = &m_vectDevSubNodeBx[i * EXTENDED + DEV_DENSITY];
			dev_u   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_U];
			dev_v   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_V];
			dev_w   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_W];

			dev_den0 = &m_vectDevSubNodeBx[i * EXTENDED + DEV_DENSITY0];
			dev_u0   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_U0];
			dev_v0   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_V0];
			dev_w0   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_VELOCITY_W0];

			dev_p   = &m_vectDevSubNodeBx[i * EXTENDED + DEV_PRESSURE];
			dev_div = &m_vectDevSubNodeBx[i * EXTENDED + DEV_DIVERGENCE];
			dev_obs = &m_vectDevSubNodeBx[i * EXTENDED + DEV_OBSTACLE];

			SolveNavierStokesEquation( DELTATIME, false, true, true, 32, 32, 32, 32, 32, 34, 34, 34 );
		}
	}

	/* 检测上次调用是否产生了错误 */
	if ( m_scHelper.GetCUDALastError( "call member function SolveNodeFlux failed", 
		__FILE__, __LINE__ ) ) 
	{
		FreeResource();
		exit(1);
	}
};


void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( not fluid->run ) return;

	if( t_totaltimes > TIMES ) 
	{
		FreeResource();
		exit(1);
	}

	printf( "%d   ", t_totaltimes );

	/* solve global */
	t_start = clock();
	SolveGlobalFlux();
	t_finish = clock();
	t_duration = (double)( t_finish - t_start ) / CLOCKS_PER_SEC;
	printf( "%f ", t_duration );

	/* solve local */
	t_start = clock();
	SolveNodeFlux();
	t_finish = clock();
	t_duration = (double)( t_finish - t_start ) / CLOCKS_PER_SEC;
	printf( "%f ", t_duration );

	GenerateVolumeData();
	RefreshStatus( fluid );

	/* FPS */
	printf( "%d", fluid->fps.uFPS );
	t_totaltimes++;
	printf("\n");
};