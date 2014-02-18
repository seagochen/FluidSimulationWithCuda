/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 18, 2014
* <File Name>     FluidSimDynamic.cu
*/

#include <iostream>
#include <cuda_runtime.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

using namespace sge;

#define DENSITY_FIELD     0
#define VELOCITY_FIELD_U  1
#define VELOCITY_FIELD_V  2
#define VELOCITY_FIELD_W  3

size_t node_size = GRIDS_X * GRIDS_X * GRIDS_X * sizeof(double);
size_t visual_size = VOLUME_X * VOLUME_X * VOLUME_X * sizeof(SGUCHAR);

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	if ( AllocateResource ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResource ();
		exit (1);
	}

	/* initialize FPS */
	InitFPS( fluid );

	/* build order */
	BuildOrder();

	/* select node */
	SelectNode(0, 0, 0);

	/* clear buffer */
	ZeroBuffers();

	/* finally, print message */
	printf( "fluid simulation ready, zero the data and preparing the stage now" );
};

void FluidSimProc::InitFPS( FLUIDSPARAM *fluid )
{
	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS = 0;
};

void FluidSimProc::BuildOrder( void )
{
	printf( "structure:\n" );

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				int index = cudaIndex3D( i, j, k, NODES_X );

				if ( index >= host_node.size() or index < 0 )
				{
					printf ( "index out of range! %s, line: %d \n", __FILE__, __LINE__ );
					exit ( 1 );
				}

				/* left */
				if ( i >= 1 )
					host_node[index]->ptrLeft = host_node[index-1];
				/* right */
				if ( i <= NODES_X - 2 )
					host_node[index]->ptrRight = host_node[index+1];
				/* down */
				if ( j >= 1 )
					host_node[index]->ptrDown = host_node[index-NODES_X];
				/* up */
				if ( j <= NODES_X - 2 )
					host_node[index]->ptrUp = host_node[index+NODES_X];
				/* back */
				if ( k >= 1 )
					host_node[index]->ptrBack = host_node[index-NODES_X*NODES_X];
				/* front */
				if ( k <= NODES_X - 2 )
					host_node[index]->ptrFront = host_node[index+NODES_X*NODES_X];

				host_node[index]->nodeIX.x = i;
				host_node[index]->nodeIX.y = j;
				host_node[index]->nodeIX.z = k;

				printf ( "no: %d | offset: %d%d%d | L: %d | R: %d | U: %d | D: %d | F: %d | B: %d \n",
					index,
					host_node[index]->nodeIX.x, 
					host_node[index]->nodeIX.y, 
					host_node[index]->nodeIX.z,
					host_node[index]->ptrLeft not_eq nullptr,
					host_node[index]->ptrRight not_eq nullptr,
					host_node[index]->ptrUp not_eq nullptr,
					host_node[index]->ptrDown not_eq nullptr,
					host_node[index]->ptrFront not_eq nullptr,
					host_node[index]->ptrBack not_eq nullptr );
			}
		}
	}

	printf( "-----------------------------------------------\n" );
};

SGRUNTIMEMSG FluidSimProc::AllocateResource ( FLUIDSPARAM *fluid )
{
	/* choose which GPU to run on, change this on a multi-GPU system. */
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		helper.CheckRuntimeErrors ( "cudaSetDevices", __FILE__, __LINE__ );

	/* allocate memory on host */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		double *ptrDens, *ptrU, *ptrV, *ptrW, *ptrObs;

		if ( helper.CreateHostBuffers( node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		/* simulation nodes */
		SimNode *node = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft = node->ptrRight = nullptr;
		node->ptrDown = node->ptrUp = nullptr;
		host_node.push_back( node );

		host_density.push_back( ptrDens );
		host_velocity_u.push_back( ptrU );
		host_velocity_v.push_back( ptrV );
		host_velocity_w.push_back( ptrW );
		host_obstacle.push_back( ptrObs );
	}

	/* allocate memory on GPU devices */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		double *ptr;
		if ( helper.CreateDeviceBuffers( node_size, 1, &ptr ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		dev_buffers.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( visual_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;
	if ( helper.CreateHostBuffers( visual_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;

	/* finally */
	return SG_RUNTIME_OK;
}  

void FluidSimProc::FreeResource ( void )
{
	/* free host resource */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		helper.FreeHostBuffers( 1, &host_density[i] );
		helper.FreeHostBuffers( 1, &host_velocity_u[i] );
		helper.FreeHostBuffers( 1, &host_velocity_v[i] );
		helper.FreeHostBuffers( 1, &host_velocity_w[i] );
		helper.FreeHostBuffers( 1, &host_obstacle[i] );
	}

	/* free device resource */
	for ( int i = 0; i < dev_buffers_num; i++ )
	{
		helper.FreeDeviceBuffers( 1, &dev_buffers[i] );
	}
	SAFE_FREE_PTR( host_visual );
	cudaFree( dev_visual );

	/* empty the vector */
	host_density.empty();
	host_velocity_u.empty();
	host_velocity_v.empty();
	host_velocity_w.empty();
	host_obstacle.empty();
	dev_buffers.empty( );
}

__global__ void kernelZeroBuffer ( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};


void FluidSimProc::ZeroBuffers ( void )
{
	cudaDeviceDim3D();

	/* zero GPU buffer first */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroBuffer <<<gridDim, blockDim>>> ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_den, node_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( host_velocity_u[i], dev_u, node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_v, node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_w, node_size, cudaMemcpyDeviceToHost );
	}
};

void FluidSimProc::NodetoDevice ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( dev_u, host_velocity_u[ix], node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_v, host_velocity_v[ix], node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_w, host_velocity_w[ix], node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_den, host_density[ix], node_size, cudaMemcpyHostToDevice  );
	cudaMemcpy( dev_obs, host_obstacle[ix], node_size, cudaMemcpyHostToDevice );
};


void FluidSimProc::DevicetoNode ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( host_velocity_u[ix], dev_u, node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_v[ix], dev_v, node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_w[ix], dev_w, node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_density[ix], dev_den, node_size, cudaMemcpyDeviceToHost  );
};

inline __host__ __device__ int atomicRand( int *seed )
{
	*seed = (69069 * *seed + 1);
	return *seed;
};

inline __host__ __device__ double atomicRandom( int *seed ) 
{
	return ( atomicRand( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double sgcrandom( int *seed )
{
	return 2.0 * ( atomicRandom( seed ) - 0.5 );
};

inline __host__ __device__ double atomicInvsqrt( double x ) 
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ double atomicSqrt( double x )
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int atomicRound( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int atomicCeil( double x )
{
	int val = atomicRound(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int atomicFloor(double x)
{
	int val = atomicRound(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

inline __host__ __device__ int atomicFabs(int value)
{
	return (value >= 0) ? value : -value;
};

inline __host__ __device__ double atomicFabs(double value)
{
	return (value >= 0.f) ? value : -value;
};

inline __host__ __device__  double atomicGetValue
	( double const *grid, int const x, int const y, int const z )
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

inline __host__ __device__  void atomicVertices
	( double *c000, double *c001, double *c011, double *c010,
	double *c100, double *c101, double *c111, double *c110,
	double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atomicGetValue ( grid, i, j, k );
	*c001 = atomicGetValue ( grid, i, j+1, k );
	*c011 = atomicGetValue ( grid, i, j+1, k+1 );
	*c010 = atomicGetValue ( grid, i, j, k+1 );
	*c100 = atomicGetValue ( grid, i+1, j, k );
	*c101 = atomicGetValue ( grid, i+1, j+1, k );
	*c111 = atomicGetValue ( grid, i+1, j+1, k+1 );
	*c110 = atomicGetValue ( grid, i+1, j, k+1 );
}

inline __host__ __device__  double atomicTrilinear
	( double const *grid, double const x, double const y, double const z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	atomicVertices ( &v000, &v001, &v011, &v010,
		&v100, &v101, &v111, &v110,
		grid, x, y, z );

	double dx = x - (int)(x);
	double dy = y - (int)(y);
	double dz = z - (int)(z);

	double c00 = v000 * ( 1 - dx ) + v001 * dx;
	double c10 = v010 * ( 1 - dx ) + v011 * dx;
	double c01 = v100 * ( 1 - dx ) + v101 * dx;
	double c11 = v110 * ( 1 - dx ) + v111 * dx;

	double c0 = c00 * ( 1 - dy ) + c10 * dy;
	double c1 = c01 * ( 1 - dy ) + c11 * dy;

	double c = c0 * ( 1 - dz ) + c1 * dz;

	return c;
};

__global__ void kernelPickData
( unsigned char *data, const double *bufs, int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data first */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;

	/* retrieve data from grid */
	double value = bufs[ Index(i, j, k) ];

	/* append data to volume data */
	int temp = atomicRound( value );
	if ( temp > 0 and temp < 250 )
		data [ cudaIndex3D(di, dj, dk, VOLUME_X) ] = (unsigned char) temp;
};

/* 采集网格数据，并转换为volumetric data */
__host__ void hostPickData( SGUCHAR *data, const double *bufs, SGINT3 *nodeIX )
{
	cudaDeviceDim3D();

	nodeIX->x *= GRIDS_X;
	nodeIX->y *= GRIDS_X;
	nodeIX->z *= GRIDS_X;

	kernelPickData cudaDevice(gridDim, blockDim)
		( data, bufs, nodeIX->x, nodeIX->y, nodeIX->z );
};

__global__ void kernelCopyBuffer
	( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer
	( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__host__ void hostSwapBuffer
	( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};


__global__ 	void kernelAddSource( double *grid, int const number )
{
	GetIndex();
	BeginSimArea();

	const int half = GRIDS_X / 2;

	switch ( number )
	{
	case 0: // density
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = 100.f;
	case 1: // velocity v
		if ( j < 3 ) 
			if ( i >= half-2 and i <= half+2 ) if ( k >= half-2 and k <= half+2 )
				grid [ Index(i,j,k) ] = 100.f;

	default: // add external force if need
		break;
	}

	EndSimArea();
};

__host__ void hostAddSource( double *dens, double *vel_u, double *vel_v, double *vel_w  )
{
	cudaDeviceDim3D();

	if ( dens != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( dens, 0 );
	if ( vel_v != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( vel_v, 1 );
};

__global__ void kernelBoundary ( double *grid, int const cd )
{
	GetIndex();
	BeginSimArea();
	// ...
	EndSimArea();
};

__global__ void kernelJacobi
( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
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
( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, cd, diffusion, divisor);
		kernelBoundary cudaDevice(gridDim, blockDim) (grid_out, cd);
	}
};

__global__ void kernelGridAdvection
( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	grid_out [ Index(i,j,k) ] = atomicTrilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__ void hostAdvection
	( double *grid_out, double const *grid_in, int const cd, 
	double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim) ( grid_out, cd );

};
#pragma endregion


__host__ void hostDiffusion
	( double *grid_out, double const *grid_in, int const cd, double const diffusion )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, cd, rate, 1+6*rate );
};


__global__ void kernelGradient
	( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
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

__global__ void kernelSubtract
	( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__ void hostProject
	( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) (div, p, vel_u, vel_v, vel_w);
	kernelBoundary cudaDevice(gridDim, blockDim) (div, DENSITY_FIELD);
	kernelBoundary cudaDevice(gridDim, blockDim) (p, DENSITY_FIELD);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, DENSITY_FIELD, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) (vel_u, vel_v, vel_w, p);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_u, VELOCITY_FIELD_U);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_v, VELOCITY_FIELD_V);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_w, VELOCITY_FIELD_W);
};

void FluidSimProc::VelocitySolver( void )
{
	hostAddSource( NULL, NULL, dev_v, NULL );

	// diffuse the velocity field (per axis):
	hostDiffusion( dev_u0, dev_u, VELOCITY_FIELD_U, VISOCITY );
	hostDiffusion( dev_v0, dev_v, VELOCITY_FIELD_V, VISOCITY );
	hostDiffusion( dev_w0, dev_w, VELOCITY_FIELD_W, VISOCITY );
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p );
	
	// advect the velocity field (per axis):
	hostAdvection( dev_u0, dev_u, VELOCITY_FIELD_U, dev_u, dev_v, dev_w );
	hostAdvection( dev_v0, dev_v, VELOCITY_FIELD_V, dev_u, dev_v, dev_w );
	hostAdvection( dev_w0, dev_w, VELOCITY_FIELD_W, dev_u, dev_v, dev_w );
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver( void )
{
	hostAddSource( dev_den, NULL, NULL, NULL );
	hostDiffusion( dev_den0, dev_den, DENSITY_FIELD, DIFFUSION );
	hostSwapBuffer( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, DENSITY_FIELD, dev_u, dev_v, dev_w );
};

void FluidSimProc::FluidSimSolver( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return;

	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				/* select node */
				SelectNode( i, j, k );
				
				/* for fluid simulation, copy the data to device */
				NodetoDevice();
				
				/* Fluid process */
				VelocitySolver();
				DensitySolver();
				
				/* Synchronize the device */
				if ( cudaDeviceSynchronize() not_eq cudaSuccess ) 
				{
					helper.CheckRuntimeErrors("cudaDeviceSynchronize failed", __FILE__, __LINE__);
					FreeResource();
					exit (1);
				}
				
				// After simulation process, retrieve data back to host, in order to 
				// avoid data flipping
				DevicetoNode();

				/* pick density */
				DensitytoVolumetric();
			}
		}
	}

	/* finally, generate volumetric image */
	GetVolumetric( fluid );
};

void FluidSimProc::DensitytoVolumetric( void )
{
	hostPickData( dev_visual, dev_den, &nPos );
}

void FluidSimProc::GetVolumetric( FLUIDSPARAM *fluid )
{
	cudaMemcpy( host_visual, dev_visual, visual_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;
};

void FluidSimProc::SelectNode( int i, int j, int k )
{
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		nPos.x = i;
		nPos.y = j;
		nPos.z = k;
	}
};

bool FluidSimProc::ActiveNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_node[ix]->active = true;
	}

	return host_node[ix]->active == true;
};

bool FluidSimProc::DeactiveNode( int i, int j, int k )
{
	int ix;
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		ix = cudaIndex3D( i, j, k, NODES_X );
		host_node[ix]->active = false;
	}

	return host_node[ix]->active == false;
};