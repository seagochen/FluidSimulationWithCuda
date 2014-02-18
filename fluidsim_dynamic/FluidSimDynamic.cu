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

inline void cudaCheckErrors ( const char* msg, const char *file, const int line )
{
	cudaError_t __err = cudaGetLastError();
	if (__err != cudaSuccess) 
	{ 
		printf ( "<<< file: %s, line %d >>> \n", file, line );
		printf ( "*error: %s \n", cudaGetErrorString(__err) );
		printf ( "%s \n", msg );
	}
};

FluidSimProc::FluidSimProc ( FLUIDSPARAM *fluid )
{
	if ( AllocateResourcePtrs ( fluid ) != SG_RUNTIME_OK )
	{
		FreeResource ();
		exit (1);
	}

	fluid->fps.dwCurrentTime = 0;
	fluid->fps.dwElapsedTime = 0;
	fluid->fps.dwFrames = 0;
	fluid->fps.dwLastUpdateTime = 0;
	fluid->fps.uFPS = 0;

	std::cout << "fluid simulation ready, zero the data and preparing the stage now" << std::endl;
	ZeroBuffers ();
};

SGRUNTIMEMSG FluidSimProc::AllocateResourcePtrs ( FLUIDSPARAM *fluid )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckErrors ( "cudaSetDevices", __FILE__, __LINE__ );

	// Allocate memory on host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static double *ptr;
		ptr = (double*) malloc ( Simul_Size * sizeof(double) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			return SG_RUNTIME_FALSE;
		}
	}

	// Allocate memory on GPU devices
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static double *ptr;
		if ( cudaMalloc( (void **) &ptr, Simul_Size * sizeof(double) ) != cudaSuccess )
		{
			cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}
	
	host_data = (GLubyte*) malloc (sizeof(GLubyte) * 
		(fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth));
	if ( cudaMalloc ((void**)&dev_data, sizeof(unsigned char) * 
		(fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth)) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMalloc failed!", __FILE__, __LINE__ );
		return SG_RUNTIME_FALSE;
	}

	// Finally
	return SG_RUNTIME_OK;
}  

void FluidSimProc::FreeResource ( void )
{
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );

	SAFE_FREE_PTR (host_data);
	cudaFree (dev_data);
}

void FluidSimProc::ZeroBuffers ( void )
{
	for ( int i = 0; i < Simul_Size; i++ )
	{
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_div [ i ] = 0.f;
		host_p [ i ] = 0.f;
	}

	if ( cudaMemcpy (dev_u, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_u0, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v0, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w0, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den0, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_div, host_div, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, host_p, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResource ();
	exit(1);

Success:
	;
}


void FluidSimProc::CopyDataToDevice ( void )
{
	if ( cudaMemcpy (dev_u, host_u, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_v, host_v, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_w, host_w, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_den, host_den, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_div, host_div, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (dev_p, host_p, sizeof(double) * Simul_Size, cudaMemcpyHostToDevice) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResource ();
	exit(1);

Success:
	;	
};


void FluidSimProc::CopyDataToHost ( void )
{
	if ( cudaMemcpy (host_u, dev_u, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_v, dev_v, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_w, dev_w, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_den, dev_den, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_div, dev_div, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;
	if ( cudaMemcpy (host_p, dev_p, sizeof(double) * Simul_Size, cudaMemcpyDeviceToHost ) != cudaSuccess )
		goto Error;

	goto Success;

Error:
	cudaCheckErrors ( "cudaMemcpy failed", __FILE__, __LINE__ );
	FreeResource ();
	exit(1);

Success:
	;	
};



inline __host__ __device__ int sgrand( int *seed )
{
	*seed = (69069 * *seed + 1);
	return *seed;
};

inline __host__ __device__ double sgrandom( int *seed ) 
{
	return ( sgrand( seed ) & 0xffff ) / (double)0x10000;
};

inline __host__ __device__  double sgcrandom( int *seed )
{
	return 2.0 * ( sgrandom( seed ) - 0.5 );
};

inline __host__ __device__ double sginvsqrt ( double x ) 
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return x;
};

inline __host__ __device__ double sgsqrt ( double x )
{
	double xhalf = 0.5f*x;
	int i = *(int*)&x;
	i = 0x5f3759df - (i>>1);
	x = *(double*)&i;
	x = x*(1.5f - xhalf*x*x);
	return 1/x;
};

inline __host__ __device__ int sground ( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

inline __host__ __device__ int sgceil ( double x )
{
	int val = sground(x);
	if (x > 0)
	{
		return (val >= (int)x) ? val : (int)x;
	}
	else
	{
		return ((int)x >= val) ? x : val;
	}
};

inline __host__ __device__  int sgfloor(double x)
{
	int val = sground(x);
	if (x > 0)
	{
		return (val < (int)x) ? val : x;
	}
	else
	{
		return ((int)x < val) ? x : val;
	}
};

inline __host__ __device__ int sgfabs(int value)
{
	return (value >= 0) ? value : -value;
};

inline __host__ __device__ double sgfabs(double value)
{
	return (value >= 0.f) ? value : -value;
};

inline __host__ __device__  double atCell (double const *grid, int const x, int const y, int const z)
{
	if ( x < gst_header ) return 0.f;
	if ( y < gst_header ) return 0.f;
	if ( z < gst_header ) return 0.f;
	if ( x > gst_tailer ) return 0.f;
	if ( y > gst_tailer ) return 0.f;
	if ( z > gst_tailer ) return 0.f;

	return grid[ Index(x,y,z) ];
};

inline __host__ __device__  void vertices (
	double *c000, double *c001, double *c011, double *c010,
	double *c100, double *c101, double *c111, double *c110,
	double const *grid, double const x, double const y, double const z )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	*c000 = atCell ( grid, i, j, k );
	*c001 = atCell ( grid, i, j+1, k );
	*c011 = atCell ( grid, i, j+1, k+1 );
	*c010 = atCell ( grid, i, j, k+1 );
	*c100 = atCell ( grid, i+1, j, k );
	*c101 = atCell ( grid, i+1, j+1, k );
	*c111 = atCell ( grid, i+1, j+1, k+1 );
	*c110 = atCell ( grid, i+1, j, k+1 );
}

inline __host__ __device__  double trilinear ( double const *grid, double const x, double const y, double const z )
{
	double v000, v001, v010, v011, v100, v101, v110, v111;
	vertices ( &v000, &v001, &v011, &v010,
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




__global__ void kernelPickData ( unsigned char *data, double const *grid )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	int temp = sground ( grid[ Index(i,j,k)] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] = (unsigned char) temp;
};

__global__ void kernelPickData ( unsigned char *data, double const *grid1, double const *grid2, double const *grid3 )
{
	GetIndex();

	data [ Index (i, j, k) ] = 0;

	// Add data from grid 1
	int temp = sground ( grid1[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	// Add data from grid 2
	temp += sground ( grid2[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;
	
	// Add data from grid 2
	temp += sground ( grid3[ Index(i,j,k) ] );
	if ( temp > 250 ) temp = 254;
	else if ( temp < 0 ) temp = 0;

	data [ Index (i, j, k) ] += (unsigned char) temp;
};


__global__ void kernelCopyBuffer ( double *grid_out, double const *grid_in )
{
	GetIndex ();

	grid_out [ Index(i,j,k) ] = grid_in [ Index(i, j, k) ];
};

__global__ void kernelSwapBuffer ( double *grid1, double *grid2 )
{
	GetIndex ();

	double temp = grid1 [ Index(i,j,k) ];
	grid1 [ Index(i,j,k) ] = grid2 [ Index(i,j,k) ];
	grid2 [ Index(i,j,k) ] = temp;
};

__global__ void kernelZeroBuffer ( double *grid )
{
	GetIndex ();
	grid [ Index(i,j,k) ] = 0.f;
};

__host__ void hostSwapBuffer ( double *grid1, double *grid2 )
{
	cudaDeviceDim3D();
	kernelSwapBuffer cudaDevice(gridDim, blockDim) (grid1, grid2);
};


#pragma region void hostAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w  )
/** 
* number:
* 0 ------ add density
* 1 ------ add velocity v
*/
__global__ 	
void kernelAddSource ( double *grid, int const number )
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

__host__
void hostAddSource ( double *dens, double *vel_u, double *vel_v, double *vel_w  )
{
	cudaDeviceDim3D();

	if ( dens != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( dens, 0 );
	if ( vel_v != NULL )
		kernelAddSource cudaDevice(gridDim, blockDim) ( vel_v, 1 );
};
#pragma endregion


/**
* cd:
* 0 -------- solve density
* 1 -------- solve velocity u
* 2 -------- solve velocity v
* 3 -------- solve velocity w
*/
__global__ void kernelBoundary ( double *grid, int const cd )
{
	GetIndex();
	BeginSimArea();
	// ...
	EndSimArea();
};


#pragma region void hostJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
__global__
void kernelJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
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

__host__
void hostJacobi ( double *grid_out, double const *grid_in, int const cd, double const diffusion, double const divisor )
{
	cudaDeviceDim3D();
	for ( int k=0; k<20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim) (grid_out, grid_in, cd, diffusion, divisor);
		kernelBoundary cudaDevice(gridDim, blockDim) (grid_out, cd);
	}
};
#pragma endregion


#pragma region void hostAdvection ( double *grid_out, double const *grid_in, int const cd, double const *u_in, double const *v_in, double const *w_in )
__global__ 
void kernelGridAdvection ( double *grid_out, double const *grid_in, double const *u_in, double const *v_in, double const *w_in )
{
	GetIndex();
	BeginSimArea();

	double u = i - u_in [ Index(i,j,k) ] * DELTATIME;
	double v = j - v_in [ Index(i,j,k) ] * DELTATIME;
	double w = k - w_in [ Index(i,j,k) ] * DELTATIME;
	grid_out [ Index(i,j,k) ] = trilinear ( grid_in, u, v, w );

	EndSimArea();
};

__host__
void hostAdvection ( double *grid_out, double const *grid_in, int const cd, double const *u_in, double const *v_in, double const *w_in )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim) ( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim) ( grid_out, cd );

};
#pragma endregion


__host__ void hostDiffusion ( double *grid_out, double const *grid_in, int const cd, double const diffusion )
{
//	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	double rate = diffusion;
	hostJacobi ( grid_out, grid_in, cd, rate, 1+6*rate );
};


#pragma region void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
__global__
void kernelGradient ( double *div, double *p, double const *vel_u, double const *vel_v, double const *vel_w )
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

__global__
void kernelSubtract ( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	GetIndex();
	BeginSimArea();

	// gradient calculated by neighbors

	vel_u [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i+1, j, k) ] - p [ Index(i-1, j, k) ] );
	vel_v [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j+1, k) ] - p [ Index(i, j-1, k) ] );
	vel_w [ Index(i, j, k) ] -= 0.5f * GRIDS_X * ( p [ Index(i, j, k+1) ] - p [ Index(i, j, k-1) ] );

	EndSimArea();
};

__host__
void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p )
{
	cudaDeviceDim3D();

	// the velocity gradient
	kernelGradient cudaDevice(gridDim, blockDim) (div, p, vel_u, vel_v, vel_w);
	kernelBoundary cudaDevice(gridDim, blockDim) (div, 0);
	kernelBoundary cudaDevice(gridDim, blockDim) (p, 0);

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, div, 0, 1.f, 6.f);

	// now subtract this gradient from our current velocity field
	kernelSubtract cudaDevice(gridDim, blockDim) (vel_u, vel_v, vel_w, p);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_u, 1);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_v, 2);
	kernelBoundary cudaDevice(gridDim, blockDim) (vel_w, 3);
};
#pragma endregion




#pragma region velocity, density, fluid simulation solver and pick data

void FluidSimProc::VelocitySolver ( void )
{
	hostAddSource ( NULL, NULL, dev_v, NULL );

	// diffuse the velocity field (per axis):
	hostDiffusion ( dev_u0, dev_u, 1, VISOCITY );
	hostDiffusion ( dev_v0, dev_v, 2, VISOCITY );
	hostDiffusion ( dev_w0, dev_w, 3, VISOCITY );
	hostSwapBuffer ( dev_u0, dev_u );
	hostSwapBuffer ( dev_v0, dev_v );
	hostSwapBuffer ( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
	
	// advect the velocity field (per axis):
	hostAdvection ( dev_u0, dev_u, 1, dev_u, dev_v, dev_w );
	hostAdvection ( dev_v0, dev_v, 2, dev_u, dev_v, dev_w );
	hostAdvection ( dev_w0, dev_w, 3, dev_u, dev_v, dev_w );
	hostSwapBuffer ( dev_u0, dev_u );
	hostSwapBuffer ( dev_v0, dev_v );
	hostSwapBuffer ( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject ( dev_u, dev_v, dev_w, dev_div, dev_p );
};

void FluidSimProc::DensitySolver ( void )
{
	hostAddSource ( dev_den, NULL, NULL, NULL );
	hostDiffusion ( dev_den0, dev_den, 0, DIFFUSION );
	hostSwapBuffer ( dev_den0, dev_den );
	hostAdvection ( dev_den, dev_den0, 0, dev_u, dev_v, dev_w );
};

void FluidSimProc::FluidSimSolver ( FLUIDSPARAM *fluid )
{
	if ( !fluid->run ) return ;

	// For fluid simulation, copy the data to device
	CopyDataToDevice();

	// Fluid process
	VelocitySolver ();
	DensitySolver ();
	PickData ( fluid );

	// Synchronize the device
	if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;

	// After simulation process, retrieve data back to host, in order to 
	// avoid data flipping
	CopyDataToHost();

	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResource ();
	exit (1);

Success:
	fluid->volume.ptrData = host_data;
};

void FluidSimProc::PickData ( FLUIDSPARAM *fluid )
{
	cudaDeviceDim3D ();
	kernelPickData  <<<gridDim, blockDim>>> ( dev_data, dev_den );

	if ( cudaMemcpy (host_data, dev_data, 
		sizeof(unsigned char) * (fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth), 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResource ();
		exit (1);
	}
};

#pragma endregion