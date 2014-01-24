/**
* <Author>      Orlando Chen
* <First>       Dec 12, 2013
* <Last>		Jan 19, 2013
* <File>        ExtFluidSimDynamic.cu
*/

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "FluidSimAreaDynamic.h"
#include "FluidMathLibDynamic.h"
#include "BufferOperationDynamic.h"

using namespace sge;
using namespace std;

struct tpnode
{
	double *ptrCurD,  *ptrCurU,*ptrCurV,  *ptrCurW, *ptrCurObs;
	double *ptrTemp,  *ptrCurP,  *ptrCurDiv;
	double *ptrLDens, *ptrLVelU, *ptrLVelV, *ptrLVelW;
	double *ptrRDens, *ptrRVelU, *ptrRVelV, *ptrRVelW;
	double *ptrUDens, *ptrUVelU, *ptrUVelV, *ptrUVelW;
	double *ptrDDens, *ptrDVelU, *ptrDVelV, *ptrDVelW;
	double *ptrFDens, *ptrFVelU, *ptrFVelV, *ptrFVelW;
	double *ptrBDens, *ptrBVelU, *ptrBVelV, *ptrBVelW;
};

static tpnode m_node;

/** 
* type:
* 0 ------ density
* 1 ------ velocity u
* 2 ------ velocity v
* 3 ------ velocity w
*/
__global__
void kernelAddSource( double *grid, double const *obstacle, int const type )
{
	GetIndex();
	int ix = Index(i,j,k);

	if ( obstacle[ ix ] eqt BD_SOURCE )
		if ( type is 0 )
			grid[ ix ] = SOURCE;
		elif ( type is 2 )
			grid[ ix ] = SOURCE;
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

/**
* cd:
* 0 -------- solve density
* 1 -------- solve velocity u
* 2 -------- solve velocity v
* 3 -------- solve velocity w
*/
#define mark_self  mark [ 0 ]
#define mark_up    mark [ 1 ]
#define mark_down  mark [ 2 ]
#define mark_left  mark [ 3 ]
#define mark_right mark [ 4 ]
#define mark_front mark [ 5 ]
#define mark_back  mark [ 6 ]
__global__ 
void kernelChecksum ( double *grid, double *checksum )
{
	GetIndex();
	if ( grid [ Index(i,j,k) ] >= 0.5f ) 
		checksum [ Index(gst_header, gst_header, gst_header) ] += grid [ Index(i,j,k) ];
};

__global__ 
void kernelBoundary ( double *grid, double *mark,
	int const cd,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	GetIndex();

	// checksum
	if ( cd eqt 0 )
	{
		if ( up [ Index(gst_header, gst_header, gst_header) ] > 0.f )
			mark_self = 1;
		else
			mark_self = 0;
	}

	grid [ Index(gst_header, j, k) ] = grid [ Index(sim_header, j, k) ];
	grid [ Index(gst_tailer, j, k) ] = grid [ Index(sim_tailer, j, k) ];
	grid [ Index(i, gst_header, k) ] = grid [ Index(i, sim_header, k) ];
	grid [ Index(i, gst_tailer, k) ] = grid [ Index(i, sim_tailer, k) ];
	grid [ Index(i, j, gst_header) ] = grid [ Index(i, j, sim_header) ];
	grid [ Index(i, j, gst_tailer) ] = grid [ Index(i, j, sim_tailer) ];

	grid [ Index(gst_header, gst_header, gst_header) ] = i0j0k0 ( grid );
	grid [ Index(gst_tailer, gst_header, gst_header) ] = i1j0k0 ( grid );
	grid [ Index(gst_header, gst_tailer, gst_header) ] = i0j1k0 ( grid );
	grid [ Index(gst_tailer, gst_tailer, gst_header) ] = i1j1k0 ( grid );
	grid [ Index(gst_header, gst_header, gst_tailer) ] = i0j0k1 ( grid );
	grid [ Index(gst_tailer, gst_header, gst_tailer) ] = i1j0k1 ( grid );
	grid [ Index(gst_header, gst_tailer, gst_tailer) ] = i0j1k1 ( grid );
	grid [ Index(gst_tailer, gst_tailer, gst_tailer) ] = i1j1k1 ( grid );

};

__host__ 
void hostBoundary ( double *grid, double *mark,
	int const cd,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	cudaDeviceDim3D();
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( up );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( down );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( left );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( right );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( front );
	kernelZeroBuffer cudaDevice(gridDim, blockDim) ( back );

	if ( cd eqt 0 )
		kernelChecksum   cudaDevice(gridDim, blockDim) ( grid, up );
	kernelBoundary cudaDevice(gridDim, blockDim) ( grid, mark, cd, 
		up, down, left, right, front, back, obstacle );
};
#undef mark_self
#undef mark_up
#undef mark_down
#undef mark_left
#undef mark_right
#undef mark_front
#undef mark_back 


__global__
void kernelJacobi ( double *grid_out,
	double const *grid_in, 
	int const cd, double const diffusion, double const divisor )
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
void hostJacobi ( double *grid_out, double *mark,
	double const *grid_in, 
	int const cd, double const diffusion, double const divisor,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	cudaDeviceDim3D();
	for ( int k = 0; k < 20; k++)
	{
		kernelJacobi cudaDevice(gridDim, blockDim)
			( grid_out, grid_in, cd, diffusion, divisor );
		hostBoundary ( grid_out, mark, cd, up, down, left, right, front, back, obstacle );
	}
};

__global__ 
void kernelGridAdvection ( double *grid_out,
	double const *grid_in, 
	double const *u_in, double const *v_in, double const *w_in )
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
void hostAdvection ( double *grid_out, double *mark,
	double const *grid_in, int const cd, 
	double const *u_in, double const *v_in, double const *w_in,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle )
{
	cudaDeviceDim3D();
	kernelGridAdvection cudaDevice(gridDim, blockDim)
		( grid_out, grid_in, u_in, v_in, w_in );
	kernelBoundary cudaDevice(gridDim, blockDim)
		( grid_out, mark, cd, up, down, left, right, front, back, obstacle );

};

__host__ void hostDiffusion ( double *grid_out, double *mark,
	double const *grid_in, int const cd, double const diffusion,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle 	)
{
	double rate = diffusion * GRIDS_X * GRIDS_X * GRIDS_X;
	hostJacobi
		( grid_out, mark,
		grid_in, cd, rate, 1 + 6 * rate, up, down, left, right, front, back, obstacle );
};

__global__
void kernelGradient ( double *div, double *p,
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

__host__
void hostGradient (  double *div, double *p,
	double const *vel_u, double const *vel_v, double const *vel_w )
{
	cudaDeviceDim3D();
	kernelGradient <<<gridDim, blockDim>>> ( div, p, vel_u, vel_v, vel_w );
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
void hostSubtract ( double *vel_u, double *vel_v, double *vel_w, double const *p )
{
	cudaDeviceDim3D();
	kernelSubtract <<<gridDim, blockDim>>> (vel_u, vel_v, vel_w, p);
};

__host__
void hostProject ( double *vel_u, double *vel_v, double *vel_w, double *div, double *p,
	double *mark,
	double *up, double *down, 
	double *left, double *right,
	double *front, double *back, 
	double const *obstacle  )
{
	// the velocity gradient
	hostGradient ( div, p, vel_u, vel_v, vel_w );
	hostBoundary ( div, mark, 0, up, down ,left, right, front, back, obstacle );
	hostBoundary ( p, mark, 0, up, down ,left, right, front, back, obstacle );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	hostJacobi(p, mark, div, 0, 1.f, 6.f, up, down ,left, right, front, back, obstacle );

	// now subtract this gradient from our current velocity field
	hostSubtract ( vel_u, vel_v, vel_w, p );
	hostBoundary ( vel_u, mark, 1, up, down ,left, right, front, back, obstacle );
	hostBoundary ( vel_v, mark, 2, up, down ,left, right, front, back, obstacle );
	hostBoundary ( vel_w, mark, 3, up, down ,left, right, front, back, obstacle );
};

#include "FunctionHelperDynamic.h"

void FluidSimProc::LinkDataset( void )
{
	/* link current grid's information */
	m_node.ptrCurD    = dev_d;
	m_node.ptrCurObs  = dev_o;
	m_node.ptrCurU    = dev_u;
	m_node.ptrCurV    = dev_v;
	m_node.ptrCurW    = dev_w;
	m_node.ptrCurP    = dev_p;
	m_node.ptrCurDiv  = dev_div;
	m_node.ptrTemp    = dev_t;

	/* link up grid */
	m_node.ptrUDens   = dev_d_U;
	m_node.ptrUVelU   = dev_u_U;
	m_node.ptrUVelV   = dev_v_U;
	m_node.ptrUVelW   = dev_w_U;

	/* link down grid */
	m_node.ptrDDens   = dev_d_D;
	m_node.ptrDVelU   = dev_u_D;
	m_node.ptrDVelV   = dev_v_D;
	m_node.ptrDVelW   = dev_w_D;

	/* link left grid */
	m_node.ptrLDens   = dev_d_L;
	m_node.ptrLVelU   = dev_u_L;
	m_node.ptrLVelV   = dev_v_L;
	m_node.ptrLVelW   = dev_w_L;

	/* link right grid */
	m_node.ptrRDens   = dev_d_R;
	m_node.ptrRVelU   = dev_u_R;
	m_node.ptrRVelV   = dev_v_R;
	m_node.ptrRVelW   = dev_w_R;

	/* link front grid */
	m_node.ptrFDens   = dev_d_F;
	m_node.ptrFVelU   = dev_u_F;
	m_node.ptrFVelV   = dev_v_F;
	m_node.ptrFVelW   = dev_w_F;

	/* link back grid */
	m_node.ptrBDens  = dev_d_B;
	m_node.ptrBVelU  = dev_u_B;
	m_node.ptrBVelV  = dev_v_B;
	m_node.ptrBVelW  = dev_w_B;
};

void FluidSimProc::VelocitySolver ( void )
{
	// diffuse the velocity field (per axis):
	hostDiffusion
		( dev_u0, dev_buf, dev_u, 1, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostDiffusion
		( dev_v0, dev_buf, dev_v, 2, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostDiffusion
		( dev_w0, dev_buf, dev_w, 3, VISOCITY, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_u0, dev_u );
	hostSwapBuffer
		( dev_v0, dev_v );
	hostSwapBuffer
		( dev_w0, dev_w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject
		( dev_u, dev_v, dev_w, dev_div, dev_p, dev_buf, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	
	// advect the velocity field (per axis):
	hostAdvection 
		( dev_u0, dev_buf, dev_u, 1, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostAdvection
		( dev_v0, dev_buf, dev_v, 2, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostAdvection
		( dev_w0, dev_buf, dev_w, 3, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_u0, dev_u );
	hostSwapBuffer
		( dev_v0, dev_v );
	hostSwapBuffer
		( dev_w0, dev_w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	hostProject
		( dev_u, dev_v, dev_w, dev_div,	dev_p, dev_buf, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
};

void FluidSimProc::DensitySolver ( void )
{
	hostDiffusion
		( dev_den0, dev_buf, dev_den, 0, DIFFUSION, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
	hostSwapBuffer
		( dev_den0, dev_den );
	hostAdvection
		( dev_den, dev_buf, dev_den0, 0, dev_u, dev_v, dev_w, dev_0, dev_1, dev_2, dev_3, dev_4, dev_5, dev_obs );
};

void FluidSimProc::PickData ( fluidsim *fluid )
{
	cudaDeviceDim3D ();
	int offseti = node_list[ IX ].i * GRIDS_X;
	int offsetj = node_list[ IX ].j * GRIDS_X;
	int offsetk = node_list[ IX ].k * GRIDS_X;

	kernelPickData cudaDevice(gridDim, blockDim) 
		( dev_visual, dev_den, offseti, offsetj, offsetk );

	size_t size = fluid->volume.uWidth * fluid->volume.uHeight * fluid->volume.uDepth;
	if ( cudaMemcpy (host_visual, dev_visual, sizeof(uchar) * size, 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ("cudaMemcpy failed", __FILE__, __LINE__);
		FreeResourcePtrs ();
		exit (1);
	}

	if ( cudaMemcpy (host_buf, dev_buf, sizeof(double) * TPBUFFER_X, 
		cudaMemcpyDeviceToHost ) != cudaSuccess )
	{
		cudaCheckErrors ( "cudaMemcpy failed",  __FILE__, __LINE__ );
		FreeResourcePtrs ();
		exit ( 1 );
	}
};

void FluidSimProc::FluidSimSolver ( fluidsim *fluid )
{
	if ( !fluid->ray.bRun ) return ;
	
	/* round robin if node is active */
	for ( int i = 0; i < host_nodes.size(); i++ )
	{
		/* active! */
		if ( host_nodes[i].bActive == true )
		{
			/* zero buffer first */
			ZeroDevData();
			
			/* for fluid simulation, copy the data to device */
			CopyDataToDevice();

			/* add source if current node is active */
			if ( i eqt 10 )
			hostAddSource ( dev_den, NULL, dev_v, NULL );
			
			/* fluid process */
			VelocitySolver ();
			DensitySolver ();
			PickData ( fluid );
			
			/* Synchronize the device */
			if ( cudaDeviceSynchronize() != cudaSuccess ) goto Error;		
			
			/* after simulation process, retrieve data back to host, in order to 
			* avoid data flipping 
			*/
			CopyDataToHost();
		}
	}
	goto Success;

Error:
	cudaCheckErrors ("cudaDeviceSynchronize failed", __FILE__, __LINE__);
	FreeResourcePtrs ();
	exit (1);

Success:
	fluid->volume.ptrData = host_visual;
};