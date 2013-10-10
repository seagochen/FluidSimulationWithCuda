#include "CFD_FuncPrototypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <SGE\SGUtils.h>

#define CUDA_Release(ptr) {if (ptr != 0) cudaFree(ptr);}
#define Error_Free(ptr0, ptr1, ptr2, ptr3) {CUDA_Release(ptr0); CUDA_Release(ptr1); CUDA_Release(ptr2); CUDA_Release(ptr3);}

#define IX(i,j) ((i)+(GridSize+2)*(j))
#define SWAP(grid0,grid) {float * tmp=grid0;grid0=grid;grid=tmp;}

__global__ void add_source_kernel ( float * grid_out, float * src_in, float dt_in )
{
	int i = threadIdx.x;
	grid_out[i] += dt_in*src_in[i];
}

void set_bnd ( int GridSize, int boundary, float * grid )
{
	int i;

	for ( i=1 ; i<=GridSize ; i++ ) {
		grid[IX(0  ,i)] = boundary==1 ? -grid[IX(1,i)] : grid[IX(1,i)];
		grid[IX(GridSize+1,i)] = boundary==1 ? -grid[IX(GridSize,i)] : grid[IX(GridSize,i)];
		grid[IX(i,0  )] = boundary==2 ? -grid[IX(i,1)] : grid[IX(i,1)];
		grid[IX(i,GridSize+1)] = boundary==2 ? -grid[IX(i,GridSize)] : grid[IX(i,GridSize)];
	}
	grid[IX(0  ,0  )] = 0.5f*(grid[IX(1,0  )]+grid[IX(0  ,1)]);
	grid[IX(0  ,GridSize+1)] = 0.5f*(grid[IX(1,GridSize+1)]+grid[IX(0  ,GridSize)]);
	grid[IX(GridSize+1,0  )] = 0.5f*(grid[IX(GridSize,0  )]+grid[IX(GridSize+1,1)]);
	grid[IX(GridSize+1,GridSize+1)] = 0.5f*(grid[IX(GridSize,GridSize+1)]+grid[IX(GridSize+1,GridSize)]);
}


void lin_solve ( int GridSize, int boundary, float * grid, float * grid0, float a, float c )
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) 
	{
		for ( i=1 ; i<=GridSize ; i++ )
		{
			for ( j=1 ; j<=GridSize ; j++ ) 
			{
				grid[IX(i,j)] = (grid0[IX(i,j)] + a*(grid[IX(i-1,j)]+grid[IX(i+1,j)]+grid[IX(i,j-1)]+grid[IX(i,j+1)]))/c;
			}
		}
		set_bnd ( GridSize, boundary, grid );
	}
}


void diffuse ( int GridSize, int boundary, float * grid, float * grid0, float diff, float dt )
{
	float a=dt*diff*GridSize*GridSize;
	lin_solve ( GridSize, boundary, grid, grid0, a, 1+4*a );
}


void advect ( int GridSize, int boundary, float * density, float * density0, float * u, float * v, float dt )
{
	int i, j, i0, j0, i1, j1;
	float grid, y, s0, t0, s1, t1, dt0;

	dt0 = dt*GridSize;
	for ( i=1 ; i<=GridSize ; i++ ) 
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			grid = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
			if (grid<0.5f) grid=0.5f; if (grid>GridSize+0.5f) grid=GridSize+0.5f; i0=(int)grid; i1=i0+1;
			if (y<0.5f) y=0.5f; if (y>GridSize+0.5f) y=GridSize+0.5f; j0=(int)y; j1=j0+1;
			s1 = grid-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
			density[IX(i,j)] = s0*(t0*density0[IX(i0,j0)]+t1*density0[IX(i0,j1)])+
				s1*(t0*density0[IX(i1,j0)]+t1*density0[IX(i1,j1)]);
		}
	}
	set_bnd ( GridSize, boundary, density );
}


void project ( int GridSize, float * u, float * v, float * p, float * div )
{
	int i, j;

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ )
		{
			div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/GridSize;		
			p[IX(i,j)] = 0;
		}
	}	
	set_bnd ( GridSize, 0, div ); set_bnd ( GridSize, 0, p );

	lin_solve ( GridSize, 0, p, div, 1, 4 );

	for ( i=1 ; i<=GridSize ; i++ )
	{
		for ( j=1 ; j<=GridSize ; j++ ) 
		{
			u[IX(i,j)] -= 0.5f*GridSize*(p[IX(i+1,j)]-p[IX(i-1,j)]);
			v[IX(i,j)] -= 0.5f*GridSize*(p[IX(i,j+1)]-p[IX(i,j-1)]);
		}
	}
	set_bnd ( GridSize, 1, u ); set_bnd ( GridSize, 2, v );
}

cudaError_t cuda_dens_step( int GridSize, float *grid, float *grid0, float *u, float *v, float diff, float dt )
{
	float *grid_cuda = 0;
	float *grid0_cuda= 0;
	float *u_cuda = 0;
	float *v_cuda = 0;

	int size = (GridSize+2)*(GridSize+2);

	cudaError_t cuda_status;

	// Choose which GPU to run on, change this on a multi-GPU system
	cuda_status = cudaSetDevice(0);
	if ( cuda_status != cudaError::cudaSuccess ) {
		ErrorMSG("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Allocate GPU buffers for those vectors
	if ( (cuda_status = cudaMalloc((void**)&grid_cuda, size * sizeof(float))) != cudaError::cudaSuccess ) 
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&grid0_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&u_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	if ( (cuda_status = cudaMalloc((void**)&v_cuda, size * sizeof(float))) != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMalloc failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Copy input vectors from host memory to GPU buffers
	// Do add_source_kernel first
	cuda_status = cudaMemcpy(grid0_cuda, grid0, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);
	if ( cuda_status != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMemcpy failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Launch a kernel on GPU with one thread for each element
	add_source_kernel<<<1, size>>>(grid_cuda, grid0_cuda, dt);

	// Check for any errors launching the kernel
	cuda_status = cudaGetLastError();
	if ( cuda_status != cudaError::cudaSuccess ) 
	{
		fprintf(stderr, "add_source_kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch
	cuda_status = cudaDeviceSynchronize();
	if ( cuda_status != cudaError::cudaSuccess )
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d (%s) after launching kernel!\n",
			cuda_status, cudaGetErrorString(cuda_status));
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	// Copy output vector from GPU buffer to host memory
	cuda_status = cudaMemcpy(grid, grid_cuda, size*sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	if ( cuda_status != cudaError::cudaSuccess )
	{
		ErrorMSG("cudaMemcpy failed!");
		Error_Free(grid_cuda, grid0_cuda, u_cuda, v_cuda);
		return cuda_status;
	}

	SWAP ( grid0, grid ); diffuse ( GridSize, 0, grid, grid0, diff, dt );
	SWAP ( grid0, grid ); advect ( GridSize, 0, grid, grid0, u, v, dt );

	return cuda_status;
};

void add_source ( int GridSize, float * grid, float * src, float dt )
{
	int i, size=(GridSize+2)*(GridSize+2);
	for ( i=0 ; i<size ; i++ ) grid[i] += dt*src[i];
}

cudaError_t cuda_vel_step( int GridSize, float * u, float * v, float * u0, float * v0, float visc, float dt )
{
	add_source ( GridSize, u, u0, dt ); add_source ( GridSize, v, v0, dt );
	SWAP ( u0, u ); diffuse ( GridSize, 1, u, u0, visc, dt );
	SWAP ( v0, v ); diffuse ( GridSize, 2, v, v0, visc, dt );
	project ( GridSize, u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( GridSize, 1, u, u0, u0, v0, dt ); advect ( GridSize, 2, v, v0, u0, v0, dt );
	project ( GridSize, u, v, u0, v0 );
};

extern "C" void dens_step( int GridSize, float *grid, float *grid0, float *u, float *v, float diff, float dt )
{
	cuda_dens_step( GridSize, grid, grid0, u, v, diff, dt );
}

extern "C" void vel_step( int GridSize, float * u, float * v, float * u0, float * v0, float visc, float dt )
{
	cuda_vel_step(GridSize,  u,  v,  u0,  v0,  visc, dt);
}