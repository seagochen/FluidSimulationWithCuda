/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 20, 2014
* <File Name>     FluidSimDynamic.cu
*/

#include <iostream>
#include <cuda_runtime.h>
#include "MacroDefinition.h"
#include "FluidSimProc.h"

using namespace sge;

#define MACRO_DENSITY     0
#define MACRO_VELOCITY_U  1
#define MACRO_VELOCITY_V  2
#define MACRO_VELOCITY_W  3
#define MACRO_SIMPLE      4

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_SOURCE     1
#define MACRO_BOUNDARY_OBSTACLE 100

#define MACRO_CENTER     0
#define MACRO_LEFT       1
#define MACRO_RIGHT      2
#define MACRO_UP         3
#define MACRO_DOWN       4
#define MACRO_FRONT      5
#define MACRO_BACK       6

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	/* initialize FPS */
	InitParams( fluid );

	/* allocate resources */
	if ( AllocateResource ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResource ();
		exit (1);
	}

	/* build order */
	BuildOrder();

	/* select node */
	ActiveNode(NODES_X/2, 0, NODES_X/2);

	/* clear buffer */
	ZeroBuffers();

	/* set boundary */
	InitBoundary();

	/* finally, print message */
	printf( "fluid simulation ready, zero the data and preparing the stage now" );
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

		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrDens ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrU ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrV ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrW ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;
		if ( helper.CreateHostBuffers( m_node_size, 1, &ptrObs ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		/* simulation nodes */
		SimNode *node = (SimNode*)malloc(sizeof(SimNode));
		node->ptrFront = node->ptrBack = nullptr;
		node->ptrLeft = node->ptrRight = nullptr;
		node->ptrDown = node->ptrUp = nullptr;
		host_node.push_back( node );

		node->active = false;

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
		if ( helper.CreateDeviceBuffers( m_node_size, 1, &ptr ) not_eq SG_RUNTIME_OK )
			return SG_RUNTIME_FALSE;

		dev_buffers.push_back(ptr);
	}

	/* allocate visual buffers */
	if ( helper.CreateDeviceBuffers( m_volm_size, 1, &dev_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;
	if ( helper.CreateHostBuffers( m_volm_size, 1, &host_visual ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;

	/* allocate temporary buffers */
	if ( helper.CreateDeviceBuffers( sizeof(double)*TPBUFFER_X, 1, &dev_tpbufs ) not_eq SG_RUNTIME_OK )
		return SG_RUNTIME_FALSE;
	if ( helper.CreateHostBuffers(sizeof(double)*TPBUFFER_X, 1, &host_tpbufs ) not_eq SG_RUNTIME_OK )
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

	/* free... */
	helper.FreeHostBuffers( 2, &host_visual, &host_tpbufs );
	helper.FreeDeviceBuffers( 2, &dev_visual, &dev_tpbufs );

	/* empty the vector */
	host_density.empty();
	host_velocity_u.empty();
	host_velocity_v.empty();
	host_velocity_w.empty();
	host_obstacle.empty();
	dev_buffers.empty( );
}

__global__ void kernelZeroBuffer( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};

__global__ void kernelZeroVisual
	( SGUCHAR *data, int const offseti, int const offsetj, int const offsetk )
{
	GetIndex();

	int di = offseti + i;
	int dj = offsetj + j;
	int dk = offsetk + k;

	/* zero data */
	data[ cudaIndex3D(di, dj, dk, VOLUME_X) ] = 0;
};

void FluidSimProc::ZeroBuffers ( void )
{
	cudaDeviceDim3D();

	/* zero GPU buffer */
	for ( int i = 0; i < dev_buffers_num; i++ )
		kernelZeroBuffer <<<gridDim, blockDim>>> ( dev_buffers[i] );

	/* zero host buffer */
	for ( int i = 0; i < NODES_X * NODES_X * NODES_X; i++ )
	{
		cudaMemcpy( host_density[i], dev_den, m_node_size, cudaMemcpyDeviceToHost);
		cudaMemcpy( host_velocity_u[i], dev_u, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_v[i], dev_v, m_node_size, cudaMemcpyDeviceToHost );
		cudaMemcpy( host_velocity_w[i], dev_w, m_node_size, cudaMemcpyDeviceToHost );
	}

	/* zero visual buffer */
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				kernelZeroVisual <<< gridDim, blockDim>>>
					( dev_visual, i * GRIDS_X, j * GRIDS_X, k * GRIDS_X );
			}
		}
	}
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
};

void FluidSimProc::NodetoDevice ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( dev_u, host_velocity_u[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_v, host_velocity_v[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_w, host_velocity_w[ix], m_node_size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_den, host_density[ix], m_node_size, cudaMemcpyHostToDevice  );
	cudaMemcpy( dev_obs, host_obstacle[ix], m_node_size, cudaMemcpyHostToDevice );
};


void FluidSimProc::DevicetoNode ( void )
{
	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	cudaMemcpy( host_velocity_u[ix], dev_u, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_v[ix], dev_v, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_velocity_w[ix], dev_w, m_node_size, cudaMemcpyDeviceToHost );
	cudaMemcpy( host_density[ix], dev_den, m_node_size, cudaMemcpyDeviceToHost  );

#if 0
	system("cls");
	printf( "no.1: %f \n", host_density[ix][Index(gst_header,gst_header,gst_header)] );
	printf( "no.2: %f \n", host_density[ix][Index(gst_tailer,gst_header,gst_header)] );
	printf( "no.3: %f \n", host_density[ix][Index(gst_tailer,gst_header,gst_tailer)] );
	printf( "no.4: %f \n", host_density[ix][Index(gst_header,gst_header,gst_tailer)] );
#endif
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

inline __host__ __device__  double atomicCrandom( int *seed )
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

__global__ void kernelCopyBuffer( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__host__ void hostSwapBuffer( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};

__device__ void atomicDensityObs( double *grids, const double *obstacle )
{
	GetIndex();
	BeginSimArea();
	/* 当前格点有障碍物，且密度大于0 */
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE and grids[Index(i,j,k)] > 0.f )
	{
		int cells  = 0;
		double val = 0; 

		if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) cells++;
		if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) cells++;

		if ( cells > 0 ) val = grids[Index(i,j,k)] / cells;
		else val = 0.f;

		if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i-1,j,k)] += val;
		if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i+1,j,k)] += val;
		if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j-1,k)] += val;
		if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j+1,k)] += val;
		if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k-1)] += val;
		if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK ) grids[Index(i,j,k+1)] += val;

		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_U( double *grids, const double *obstacle )
{
	GetIndex();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i-1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i-1,j,k)] = grids[Index(i-1,j,k)] -  grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i+1,j,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i+1,j,k)] = grids[Index(i+1,j,k)] -  grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_V( double *grids, const double *obstacle )
{
	GetIndex();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i,j-1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j-1,k)] = grids[Index(i,j-1,k)] - grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i,j+1,k)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j+1,k)] = grids[Index(i,j+1,k)] - grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__device__ void atomicVelocityObs_W( double *grids, const double *obstacle )
{
	GetIndex();
	BeginSimArea();
	if ( obstacle[Index(i,j,k)] eqt MACRO_BOUNDARY_OBSTACLE )
	{
		if ( grids[Index(i,j,k)] > 0.f )
		{
			if ( obstacle[Index(i,j,k-1)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j,k-1)] = grids[Index(i,j,k-1)] - grids[Index(i,j,k)];
		}
		else
		{
			if ( obstacle[Index(i,j,k+1)] eqt MACRO_BOUNDARY_BLANK )
				grids[Index(i,j,k+1)] = grids[Index(i,j,k+1)] - grids[Index(i,j,k)];
		}
		grids[Index(i,j,k)] = 0.f;
	}
	EndSimArea();
};

__global__ void kernelObstacle( double *grids, const double *obstacle, const int field )
{
	switch( field )
	{
	case MACRO_DENSITY:
		atomicDensityObs( grids, obstacle );
		break;

	case MACRO_VELOCITY_U:
		atomicVelocityObs_U( grids, obstacle );
		break;

	case MACRO_VELOCITY_V:
		atomicVelocityObs_V( grids, obstacle );
		break;

	case MACRO_VELOCITY_W:
		atomicVelocityObs_W( grids, obstacle );
		break;

	default:
		break;
	}
};

__global__ void kernelJacobi( double *grid_out, double const *grid_in, double const diffusion, double const divisor )
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
	( double *grid_out, double const *grid_in,
	double const *obstacle, int const field, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, diffusion, divisor);
	}
	kernelObstacle cudaDevice(gridDim, blockDim) ( grid_out, obstacle, field );
};

__global__ void kernelGridAdvection( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;

//	if ( u >= 0.f and u < GRIDS_X and v >= 0.f and v < GRIDS_X and v >= 0.f and v < GRIDS_X )
		grid_out [ Index(i,j,k) ] = atomicTrilinear ( grid_in, u, v, w );
//	else
//		grid_out [ Index(i,j,k) ] = 0.f;

	EndSimArea();
};

__host__ void hostAdvection
	( double *grid_out, double const *grid_in,
	double const *obstacle, int const field,
	double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelObstacle cudaDevice(gridDim, blockDim) ( grid_out, obstacle, field );
};

__host__ void hostDiffusion
	( double *grid_out, double const *grid_in, double const diffusion,
	double const *obstacle, int const field )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, obstacle, field, rate, 1+6*rate );
};


__global__ void kernelGradient( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
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

__host__ void hostProject( double *vel_u, double *vel_v, double *vel_w, double *div, double *p, double const *obs )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) ( div, p, vel_u, vel_v, vel_w );
	kernelObstacle cudaDevice(gridDim, blockDim) ( div, obs, MACRO_SIMPLE );
	kernelObstacle cudaDevice(gridDim, blockDim) ( p, obs, MACRO_SIMPLE );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, obs, MACRO_SIMPLE, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) ( vel_u, vel_v, vel_w, p );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_u, obs, MACRO_VELOCITY_U );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_v, obs, MACRO_VELOCITY_V );
	kernelObstacle cudaDevice(gridDim, blockDim) ( vel_w, obs, MACRO_VELOCITY_W );
};

void FluidSimProc::VelocitySolver( void )
{
	// diffuse the velocity field (per axis):
	hostDiffusion( dev_u0, dev_u, VISOCITY, dev_obs, MACRO_VELOCITY_U );
	hostDiffusion( dev_v0, dev_v, VISOCITY, dev_obs, MACRO_VELOCITY_V );
	hostDiffusion( dev_w0, dev_w, VISOCITY, dev_obs, MACRO_VELOCITY_W );
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
	
	// advect the velocity field (per axis):
	hostAdvection( dev_u0, dev_u, dev_obs, MACRO_VELOCITY_U, dev_u, dev_v, dev_w );
	hostAdvection( dev_v0, dev_v, dev_obs, MACRO_VELOCITY_V, dev_u, dev_v, dev_w );
	hostAdvection( dev_w0, dev_w, dev_obs, MACRO_VELOCITY_W, dev_u, dev_v, dev_w );
	hostSwapBuffer( dev_u0, dev_u );
	hostSwapBuffer( dev_v0, dev_v );
	hostSwapBuffer( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject( dev_u, dev_v, dev_w, dev_div, dev_p, dev_obs );
};

void FluidSimProc::DensitySolver( void )
{
	hostDiffusion( dev_den0, dev_den, DIFFUSION, dev_obs, MACRO_DENSITY );
	hostSwapBuffer( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, dev_obs, MACRO_DENSITY, dev_u, dev_v, dev_w );
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
				if ( SelectNode( i, j, k ) )
				{
					/* for fluid simulation, copy the data to device */
					NodetoDevice();
					
					/* Fluid process */
					AddSource();
					VelocitySolver();
					DensitySolver();
					
					/* retrieve data back to host */
					DevicetoNode();

					/* pick density */
					DensitytoVolumetric();
				}
			}
		}
	}
	/* finally, generate volumetric image */
	GetVolumetric( fluid );

	/* tracing */
	TracingDensity();
};

void FluidSimProc::DensitytoVolumetric( void )
{
	hostPickData( dev_visual, dev_den, &nPos );
}

void FluidSimProc::GetVolumetric( FLUIDSPARAM *fluid )
{
	cudaMemcpy( host_visual, dev_visual, m_volm_size, cudaMemcpyDeviceToHost );
	fluid->volume.ptrData = host_visual;
};

bool FluidSimProc::SelectNode( int i, int j, int k )
{
	if ( i >= 0 and i < NODES_X and j >= 0 and j < NODES_X and k >= 0 and k < NODES_X )
	{
		nPos.x = i;
		nPos.y = j;
		nPos.z = k;
	}

	int ix = cudaIndex3D( i, j, k, NODES_X );
	
	return host_node[ix]->active;
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

__global__ void kernelAddSource
	( double *density, double *vel_u, double *vel_v, double *vel_w, double *obs )
{
	GetIndex();

	const int half = GRIDS_X / 2;

	if ( obs[ Index(i,j,k) ] eqt MACRO_BOUNDARY_SOURCE )
	{
		/* add source to grids */
		density[Index(i,j,k)] = SOURCE_DENSITY;

		/* add velocity to grids */
		if ( i < half )
			vel_u[Index(i,j,k)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
		elif( i >= half )
			vel_u[Index(i,j,k)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;

		vel_v[Index(i,j,k)] = SOURCE_VELOCITY;

		if ( k < half )
			vel_w[Index(i,j,k)] = -SOURCE_VELOCITY * DELTATIME * DELTATIME;
		elif ( k >= half )
			vel_w[Index(i,j,k)] =  SOURCE_VELOCITY * DELTATIME * DELTATIME;
	}
};

/* add source */
void FluidSimProc::AddSource( void )
{
	cudaDeviceDim3D();

	kernelAddSource<<<gridDim, blockDim>>> ( dev_den, dev_u, dev_v, dev_w, dev_obs );
};

__global__ void kernelSetBoundary( double *grids )
{
	GetIndex();
	
	const int half = GRIDS_X / 2;
	
	if ( j < 3 and i >= half-2 and i <= half+2 and k >= half-2 and k <= half+2 )
		grids[ Index(i,j,k) ] = MACRO_BOUNDARY_SOURCE;
};

/* initialize boundary condition */
void FluidSimProc::InitBoundary( void )
{
	cudaDeviceDim3D();

	/* zero boundary buffers */
	kernelZeroBuffer<<<gridDim, blockDim>>>( dev_obs );

	for ( int i = 0; i < host_obstacle.size(); i++ )
	{
		if ( cudaMemcpy( host_obstacle[i], dev_obs,
			m_node_size, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
	}

	/* select middle node */
	SelectNode(NODES_X/2, 0, NODES_X/2);

	const int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );

	/* set boundary */
	kernelSetBoundary<<<gridDim, blockDim>>>( dev_obs );
	
	if ( cudaMemcpy( host_obstacle[ix], dev_obs,
		m_node_size, cudaMemcpyDeviceToHost) not_eq cudaSuccess )
	{
		helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
		FreeResource();
		exit( 1 );
	}
};

#if 0
__global__ void kernelFloodBoundary( double *grids )
{
	GetIndex();

	/* faces */
	grids[Index(gst_header,j,k)] = grids[Index(sim_header,j,k)];
	grids[Index(gst_tailer,j,k)] = grids[Index(sim_tailer,j,k)];
	grids[Index(i,gst_header,k)] = grids[Index(i,sim_header,k)];
	grids[Index(i,gst_tailer,k)] = grids[Index(i,sim_tailer,k)];
	grids[Index(i,j,gst_header)] = grids[Index(i,j,sim_header)];
	grids[Index(i,j,gst_tailer)] = grids[Index(i,j,sim_tailer)];

	/* edges */
	grids[Index(i,gst_header,gst_header)] = ( grids[Index(i,sim_header,gst_header)] + grids[Index(i,gst_header,sim_header)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_header)] = ( grids[Index(i,sim_tailer,gst_header)] + grids[Index(i,gst_tailer,sim_header)] ) / 2.f;
	grids[Index(i,gst_header,gst_tailer)] = ( grids[Index(i,sim_header,gst_tailer)] + grids[Index(i,gst_header,sim_tailer)] ) / 2.f;
	grids[Index(i,gst_tailer,gst_tailer)] = ( grids[Index(i,sim_tailer,gst_tailer)] + grids[Index(i,gst_tailer,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,j,gst_header)] = ( grids[Index(sim_header,j,gst_header)] + grids[Index(gst_header,j,sim_header)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_header)] = ( grids[Index(sim_tailer,j,gst_header)] + grids[Index(gst_tailer,j,sim_header)] ) / 2.f;
	grids[Index(gst_header,j,gst_tailer)] = ( grids[Index(sim_header,j,gst_tailer)] + grids[Index(gst_header,j,sim_tailer)] ) / 2.f;
	grids[Index(gst_tailer,j,gst_tailer)] = ( grids[Index(sim_tailer,j,gst_tailer)] + grids[Index(gst_tailer,j,sim_tailer)] ) / 2.f;

	grids[Index(gst_header,gst_header,k)] = ( grids[Index(sim_header,gst_header,k)] + grids[Index(gst_header,sim_header,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_header,k)] = ( grids[Index(sim_tailer,gst_header,k)] + grids[Index(gst_tailer,sim_header,k)] ) / 2.f;
	grids[Index(gst_header,gst_tailer,k)] = ( grids[Index(sim_header,gst_tailer,k)] + grids[Index(gst_header,sim_tailer,k)] ) / 2.f;
	grids[Index(gst_tailer,gst_tailer,k)] = ( grids[Index(sim_tailer,gst_tailer,k)] + grids[Index(gst_tailer,sim_tailer,k)] ) / 2.f;

	/* vetices */
	grids[Index(gst_header,gst_header,gst_header)] = ( grids[Index(sim_header,gst_header,gst_header)] + grids[Index(gst_header,sim_header,gst_header)] + grids[Index(gst_header,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_header,gst_tailer)] = ( grids[Index(sim_header,gst_header,gst_tailer)] + grids[Index(gst_header,sim_header,gst_tailer)] + grids[Index(gst_header,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_header)] = ( grids[Index(sim_header,gst_tailer,gst_header)] + grids[Index(gst_header,sim_tailer,gst_header)] + grids[Index(gst_header,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_header,gst_tailer,gst_tailer)] = ( grids[Index(sim_header,gst_tailer,gst_tailer)] + grids[Index(gst_header,sim_tailer,gst_tailer)] + grids[Index(gst_header,gst_tailer,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_header)] = ( grids[Index(sim_tailer,gst_header,gst_header)] + grids[Index(gst_tailer,sim_header,gst_header)] + grids[Index(gst_tailer,gst_header,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_header,gst_tailer)] = ( grids[Index(sim_tailer,gst_header,gst_tailer)] + grids[Index(gst_tailer,sim_header,gst_tailer)] + grids[Index(gst_tailer,gst_header,sim_tailer)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_header)] = ( grids[Index(sim_tailer,gst_tailer,gst_header)] + grids[Index(gst_tailer,sim_tailer,gst_header)] + grids[Index(gst_tailer,gst_tailer,sim_header)] ) / 3.f;
	grids[Index(gst_tailer,gst_tailer,gst_tailer)] = ( grids[Index(sim_tailer,gst_tailer,gst_tailer)] + grids[Index(gst_tailer,sim_tailer,gst_tailer)] + grids[Index(gst_tailer,gst_tailer,sim_tailer)]) / 3.f;
};
#endif

void FluidSimProc::TracingDensity( void )
{
	for ( int i = 0; i < NODES_X; i++ )
	{
		for ( int j = 0; j < NODES_X; j++ )
		{
			for ( int k = 0; k < NODES_X; k++ )
			{
				if ( SelectNode( i, j, k ) )
				{
					DataFlooding( host_density, true );
					DataFlooding( host_velocity_u, false );
					DataFlooding( host_velocity_v, false );
					DataFlooding( host_velocity_w, false );
				}
			}
		}
	}
};

__device__ void atomicHandleFaceLR( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
	
	/* copy entire face from center node */
	grids[Index(header,j,k)] = center[Index(tailer,j,k)];

	/* wait, we have left something which may cause un-conservation of mass */
	grids[Index(header,sim_header,k)] = center[Index(tailer,sim_header,k)] / 2.f;
	grids[Index(header,sim_tailer,k)] = center[Index(tailer,sim_tailer,k)] / 2.f;
	grids[Index(header,j,sim_header)] = center[Index(tailer,j,sim_header)] / 2.f;
	grids[Index(header,j,sim_tailer)] = center[Index(tailer,j,sim_tailer)] / 2.f;

	grids[Index(header,sim_header,sim_header)] = center[Index(tailer,sim_header,sim_header)] / 3.f;
	grids[Index(header,sim_header,sim_tailer)] = center[Index(tailer,sim_header,sim_tailer)] / 3.f;
	grids[Index(header,sim_tailer,sim_header)] = center[Index(tailer,sim_tailer,sim_header)] / 3.f;
	grids[Index(header,sim_tailer,sim_tailer)] = center[Index(tailer,sim_tailer,sim_tailer)] / 3.f;
};
 
__device__ void atomicHandleFaceFB( double *grids, double const *center, int header, int tailer )
{
	GetIndex();

	grids[Index(i,header,k)] = center[Index(i,tailer,k)];

	grids[Index(sim_header,header,k)] = center[Index(sim_header,tailer,k)] / 2.f;
	grids[Index(sim_tailer,header,k)] = center[Index(sim_tailer,tailer,k)] / 2.f;
	grids[Index(i,header,sim_header)] = center[Index(i,tailer,sim_header)] / 2.f;
	grids[Index(i,header,sim_tailer)] = center[Index(i,tailer,sim_tailer)] / 2.f;

	grids[Index(sim_header,header,sim_header)] = center[Index(sim_header,tailer,sim_header)] / 2.f;
	grids[Index(sim_tailer,header,sim_header)] = center[Index(sim_tailer,tailer,sim_header)] / 2.f;
	grids[Index(sim_header,header,sim_tailer)] = center[Index(sim_header,tailer,sim_tailer)] / 2.f;
	grids[Index(sim_tailer,header,sim_tailer)] = center[Index(sim_tailer,tailer,sim_tailer)] / 2.f;
};

__device__ void atomicHandleFaceUD( double *grids, double const *center, int header, int tailer )
{
	GetIndex();
 
	grids[Index(i,j,header)] = center[Index(i,j,tailer)];
 
	grids[Index(sim_header,j,header)] = center[Index(sim_header,j,tailer)] / 2.f;
	grids[Index(i,sim_header,header)] = center[Index(i,sim_header,tailer)] / 2.f;
	grids[Index(sim_tailer,j,header)] = center[Index(sim_tailer,j,tailer)] / 2.f;
	grids[Index(i,sim_tailer,header)] = center[Index(i,sim_tailer,tailer)] / 2.f;

	grids[Index(sim_header,sim_header,header)] = center[Index(sim_header,sim_header,tailer)] / 3.f;
	grids[Index(sim_header,sim_tailer,header)] = center[Index(sim_header,sim_tailer,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_header,header)] = center[Index(sim_tailer,sim_header,tailer)] / 3.f;
	grids[Index(sim_tailer,sim_tailer,header)] = center[Index(sim_tailer,sim_tailer,tailer)] / 3.f;
};

__global__ void kernelFloodBuffersBetweenNodes( double *grids, double *center, const int ops )
{
	switch ( ops )
 	{
	case MACRO_LEFT:
		atomicHandleFaceLR( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_RIGHT:
		atomicHandleFaceLR( grids, center, sim_header, sim_tailer );
		break;

	case MACRO_UP:
		atomicHandleFaceUD( grids, center, sim_header, sim_tailer );
		break;

	case MACRO_DOWN:
		atomicHandleFaceUD( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_FRONT:
		atomicHandleFaceFB( grids, center, sim_tailer, sim_header );
		break;

	case MACRO_BACK:
		atomicHandleFaceFB( grids, center, sim_header, sim_tailer );
		break;

	default:
		break;
	}

	GetIndex();

	center[Index(gst_header,j,k)] = grids[Index(gst_header,j,k)] = 0.f;
	center[Index(gst_tailer,j,k)] = grids[Index(gst_tailer,j,k)] = 0.f;
	center[Index(i,gst_header,k)] = grids[Index(i,gst_header,k)] = 0.f;
	center[Index(i,gst_tailer,k)] = grids[Index(i,gst_tailer,k)] = 0.f;
	center[Index(i,j,gst_header)] = grids[Index(i,j,gst_header)] = 0.f;
	center[Index(i,j,gst_tailer)] = grids[Index(i,j,gst_tailer)] = 0.f;
};

__global__ void kernelSumDensity( double *grids, double *buffs, const int ops )
{
	GetIndex();

#define num_dens_center  buffs[ 0 ]
#define num_dens_left    buffs[ 1 ]
#define num_dens_right   buffs[ 2 ]
#define num_dens_up      buffs[ 3 ]
#define num_dens_down    buffs[ 4 ]
#define num_dens_front   buffs[ 5 ]
#define num_dens_back    buffs[ 6 ]
 	
	if ( grids[Index(i,j,k)] > 0.f )
 	{
		switch ( ops )
		{
		case MACRO_LEFT:
			num_dens_left += grids[Index(i,j,k)];
			break;
		case MACRO_RIGHT:
			num_dens_right += grids[Index(i,j,k)];
			break;
		case MACRO_UP:
			num_dens_up += grids[Index(i,j,k)];
			break;
		case MACRO_DOWN:
			num_dens_down += grids[Index(i,j,k)];
			break;
		case MACRO_FRONT:
			num_dens_front += grids[Index(i,j,k)];
			break;
		case MACRO_BACK:
			num_dens_back += grids[Index(i,j,k)];
			break;
		case MACRO_CENTER:
			num_dens_center += grids[Index(i,j,k)];
			break;

		default:
			break;
		}
 	}
#undef num_dens_center
#undef num_dens_left  
#undef num_dens_right 
#undef num_dens_up    
#undef num_dens_down  
#undef num_dens_front 
#undef num_dens_back 

};

void FluidSimProc::DataFlooding( vector<double*> container, bool bDens )
{
	cudaDeviceDim3D();
	
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_center );
 	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_front );
 	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_back );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_left );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_right );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_up );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( dev_down );
 
 	int ix = cudaIndex3D( nPos.x, nPos.y, nPos.z, NODES_X );
	cudaMemcpy( dev_center, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	
 	if ( nPos.x - 1 >= 0 ) // left
 	{
		ix = cudaIndex3D( nPos.x-1, nPos.y, nPos.z, NODES_X );
		cudaMemcpy( dev_left, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.x + 1 < NODES_X ) // right
 	{
 		ix = cudaIndex3D( nPos.x + 1, nPos.y, nPos.z, NODES_X );
		cudaMemcpy( dev_right, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.y - 1 >= 0 ) // down
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y - 1, nPos.z, NODES_X );
		cudaMemcpy( dev_down, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.y + 1 < NODES_X ) // up
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y + 1, nPos.z, NODES_X );
		cudaMemcpy( dev_up, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.z - 1 >= 0 ) // back
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z - 1, NODES_X );
		cudaMemcpy( dev_back, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}
 	if ( nPos.z + 1 < NODES_X ) // front
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z + 1, NODES_X );
		cudaMemcpy( dev_front, container[ix], m_node_size, cudaMemcpyHostToDevice );
 	}

	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_left, dev_center, MACRO_LEFT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_right, dev_center, MACRO_RIGHT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_up, dev_center, MACRO_UP );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_down, dev_center, MACRO_DOWN );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_front, dev_center, MACRO_FRONT );
	kernelFloodBuffersBetweenNodes <<<gridDim,blockDim>>> ( dev_back, dev_center, MACRO_BACK );

	if ( bDens )
 	{
		for ( int i = 0; i < TPBUFFER_X; i++)
		{
			host_tpbufs[i] = 0.f;
		}
		if ( cudaMemcpy( dev_tpbufs, host_tpbufs, 
			sizeof(double) * TPBUFFER_X, cudaMemcpyHostToDevice ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}

		kernelSumDensity<<<gridDim, blockDim>>> ( dev_center, dev_tpbufs, MACRO_CENTER );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_left, dev_tpbufs, MACRO_LEFT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_right, dev_tpbufs, MACRO_RIGHT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_up, dev_tpbufs, MACRO_UP );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_down, dev_tpbufs, MACRO_DOWN );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_front, dev_tpbufs, MACRO_FRONT );
		kernelSumDensity<<<gridDim, blockDim>>> ( dev_back, dev_tpbufs, MACRO_BACK );
		
		if ( cudaMemcpy( host_tpbufs, dev_tpbufs, sizeof(double) * TPBUFFER_X, cudaMemcpyDeviceToHost ) not_eq cudaSuccess )
		{
			helper.CheckRuntimeErrors( "cudaMemcpy failed", __FILE__, __LINE__ );
			FreeResource();
			exit( 1 );
		}
		
#define num_dens_center  host_tpbufs[ 0 ]
#define num_dens_left    host_tpbufs[ 1 ]
#define num_dens_right   host_tpbufs[ 2 ]
#define num_dens_up      host_tpbufs[ 3 ]
#define num_dens_down    host_tpbufs[ 4 ]
#define num_dens_front   host_tpbufs[ 5 ]
#define num_dens_back    host_tpbufs[ 6 ]

#if 0
		system( "cls" );
		printf( "center:  %f\n", num_dens_center );
		printf( "left:    %f\n", num_dens_left );
		printf( "right:   %f\n", num_dens_right );
		printf( "up:      %f\n", num_dens_up );
		printf( "down:    %f\n", num_dens_down );
		printf( "front:   %f\n", num_dens_front );
		printf( "back:    %f\n", num_dens_back );
#endif
 	}

	/* retrieve data back to node if density filled */
	if ( nPos.x - 1 >= 0 and num_dens_left > 0.f ) // left
 	{
		ix = cudaIndex3D( nPos.x-1, nPos.y, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_left, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.x + 1 < NODES_X and num_dens_right > 0.f ) // right
 	{
 		ix = cudaIndex3D( nPos.x + 1, nPos.y, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_right, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.y - 1 >= 0 and num_dens_down > 0.f ) // down
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y - 1, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_down, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.y + 1 < NODES_X and num_dens_up > 0.f ) // up
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y + 1, nPos.z, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_up, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.z - 1 >= 0 and num_dens_back > 0.f ) // back
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z - 1, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_back, m_node_size, cudaMemcpyDeviceToHost );
 	}
	if ( nPos.z + 1 < NODES_X and num_dens_front > 0.f ) // front
 	{
 		ix = cudaIndex3D( nPos.x, nPos.y, nPos.z + 1, NODES_X );
		host_node[ix]->active = true;
		cudaMemcpy( container[ix], dev_front, m_node_size, cudaMemcpyDeviceToHost );
 	}

#undef num_dens_center
#undef num_dens_left  
#undef num_dens_right 
#undef num_dens_up    
#undef num_dens_down  
#undef num_dens_front 
#undef num_dens_back
};