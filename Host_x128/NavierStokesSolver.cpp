/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Apr 03, 2014
* <File Name>     NavierStokesSolver.cpp
*/

#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"
#include "ISO646.h"


using namespace sge;

int IX( int i, int j, int k, int tx, int ty ) { return k * tx * ty + j * tx + i; };

double atomicGetValue
	( cdouble *grid, cint x, cint y, cint z, cint tx, cint ty, cint tz )
{
	if ( x < 0 or x >= tx ) return 0.f;
	if ( y < 0 or y >= ty ) return 0.f;
	if ( z < 0 or z >= tz ) return 0.f;

	return grid[ IX( x, y, z, tx, ty ) ];
};

double atomicTrilinear
	( cdouble *grid, cdouble x, cdouble y, cdouble z, cint tx, cint ty, cint tz )
{
	int i = (int)x;
	int j = (int)y;
	int k = (int)z;

	double v000 = atomicGetValue( grid, i, j, k, tx, ty, tz );
	double v001 = atomicGetValue( grid, i, j+1, k, tx, ty, tz );
	double v011 = atomicGetValue( grid, i, j+1, k+1, tx, ty, tz );
	double v010 = atomicGetValue( grid, i, j, k+1, tx, ty, tz );
	double v100 = atomicGetValue( grid, i+1, j, k, tx, ty, tz );
	double v101 = atomicGetValue( grid, i+1, j+1, k, tx, ty, tz );
	double v111 = atomicGetValue( grid, i+1, j+1, k+1, tx, ty, tz );
	double v110 = atomicGetValue( grid, i+1, j, k+1, tx, ty, tz );

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


void FluidSimProc::Advection( double *out, cdouble *in, cdouble *u, cdouble *v, cdouble *w, cdouble dt )
{
	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
		double velu = i - u[ ix(i,j,k) ] * dt;
		double velv = j - v[ ix(i,j,k) ] * dt;
		double velw = k - w[ ix(i,j,k) ] * dt;

		out[ ix(i,j,k) ] = atomicTrilinear( in, velu, velv, velw, 128, 128, 128 );
	}
};




#if 0
void add_source ( int N, float * x, float * s, float dt )
{
	int i, size=(N+2)*(N+2);
	for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
}

void set_bnd ( int N, int b, float * x )
{
	int i;

	for ( i=1 ; i<=N ; i++ ) {
		x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
		x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
		x[IX(i,0  )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
		x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
	}
	x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
	x[IX(0  ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0  ,N)]);
	x[IX(N+1,0  )] = 0.5f*(x[IX(N,0  )]+x[IX(N+1,1)]);
	x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N)]);
}

void lin_solve ( int N, int b, float * x, float * x0, float a, float c )
{
	int i, j, k;

	for ( k=0 ; k<20 ; k++ ) {
		FOR_EACH_CELL
			x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
		END_FOR
		set_bnd ( N, b, x );
	}
}

void diffuse ( int N, int b, float * x, float * x0, float diff, float dt )
{
	float a=dt*diff*N*N;
	lin_solve ( N, b, x, x0, a, 1+4*a );
}

void advect ( int N, int b, float * d, float * d0, float * u, float * v, float dt )
{
	int i, j, i0, j0, i1, j1;
	float x, y, s0, t0, s1, t1, dt0;

	dt0 = dt*N;
	FOR_EACH_CELL
		x = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
		if (x<0.5f) x=0.5f; if (x>N+0.5f) x=N+0.5f; i0=(int)x; i1=i0+1;
		if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
		s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
		d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+
					 s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
	END_FOR
	set_bnd ( N, b, d );
}

void project ( int N, float * u, float * v, float * p, float * div )
{
	int i, j;

	FOR_EACH_CELL
		div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/N;
		p[IX(i,j)] = 0;
	END_FOR	
	set_bnd ( N, 0, div ); set_bnd ( N, 0, p );

	lin_solve ( N, 0, p, div, 1, 4 );

	FOR_EACH_CELL
		u[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)]-p[IX(i-1,j)]);
		v[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)]-p[IX(i,j-1)]);
	END_FOR
	set_bnd ( N, 1, u ); set_bnd ( N, 2, v );
}

void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt )
{
	add_source ( N, x, x0, dt );
	SWAP ( x0, x ); diffuse ( N, 0, x, x0, diff, dt );
	SWAP ( x0, x ); advect ( N, 0, x, x0, u, v, dt );
}

void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt )
{
	add_source ( N, u, u0, dt ); add_source ( N, v, v0, dt );
	SWAP ( u0, u ); diffuse ( N, 1, u, u0, visc, dt );
	SWAP ( v0, v ); diffuse ( N, 2, v, v0, visc, dt );
	project ( N, u, v, u0, v0 );
	SWAP ( u0, u ); SWAP ( v0, v );
	advect ( N, 1, u, u0, u0, v0, dt ); advect ( N, 2, v, v0, u0, v0, dt );
	project ( N, u, v, u0, v0 );
}

#endif 











void FluidSimProc::Jacobi( double *out, cdouble *in, cdouble diff, cdouble divisor )
{
	double dix = ( divisor > 0 ) ? divisor : 1.f;

	for ( int n = 0; n < 20; n++ )
	{
		for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
		{
			out[ ix(i,j,k) ] = ( in[ ix(i,j,k) ] + diff * (
				out[ ix(i-1,j,k) ] + out[ ix(i+1,j,k) ] +
				out[ ix(i,j-1,k) ] + out[ ix(i,j+1,k) ] +
				out[ ix(i,j,k-1) ] + out[ ix(i,j,k+1) ]
				) ) / dix;
		}
	}
};

//void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
//{
//	double rate = diff * 128 * 128 * 128;
//	Jacobi( out, in, rate, 1+6*rate );
//};

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
//	for ( int n = 0; n < 20; n++ )
//	{
//		for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
//		{
//			out[ ix(i,j,k) ] = ( in[ ix(i,j,k) ] + diff * (
//				out[ ix(i-1,j,k) ] + out[ ix(i+1,j,k) ] +
//				out[ ix(i,j-1,k) ] + out[ ix(i,j+1,k) ] +
//				out[ ix(i,j,k-1) ] + out[ ix(i,j,k+1) ]
//				) ) / dix;
//		}
//	}

	double x = 0.f;

	double dix = (diff > 0.f) ? diff : 1.f;

	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
		x = ( in[ ix(i-1,j,k) ] + in[ ix(i+1,j,k) ] + 
			in[ ix(i,j-1,k) ] + in[ ix(i,j+1,k) ] +
			in[ ix(i,j,k-1) ] + in[ ix(i,j,k+1) ] );

		out[ix(i,j,k)] = (x + in[ix(i,j,k)] / dix) / (6+dix);
	}
};

void kernelGradient( double *div, double *prs, cdouble *u, cdouble *v, cdouble *w )
{
	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
		cdouble hx = 1.f / (double)128;
		cdouble hy = 1.f / (double)128;
		cdouble hz = 1.f / (double)128;

		// previous instantaneous magnitude of velocity gradient 
		//		= (sum of velocity gradients per axis)/2N:
		div[ IX(i,j,k,128,128) ] = -0.5f * (
			hx * ( u[ IX(i+1,j,k,128,128) ] - u[ IX(i-1,j,k,128,128) ] ) +
			hy * ( v[ IX(i,j+1,k,128,128) ] - v[ IX(i,j-1,k,128,128) ] ) +
			hz * ( w[ IX(i,j,k+1,128,128) ] - w[ IX(i,j,k-1,128,128) ] ) );

		// zero out the present velocity gradient
		prs[ IX(i,j,k,128,128) ] = 0.f;
	}
};


void kernelSubtract( double *u, double *v, double *w, double *prs )
{
	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
		u[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i+1,j,k,128,128) ] - prs[ IX(i-1,j,k,128,128) ] );
		v[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i,j+1,k,128,128) ] - prs[ IX(i,j-1,k,128,128) ] );
		w[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i,j,k+1,128,128) ] - prs[ IX(i,j,k-1,128,128) ] );
	}
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
	// the velocity gradient
	kernelGradient( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
//	Jacobi( p, div, 1.f, 6.f );



	// now subtract this gradient from our current velocity field
	kernelSubtract ( u, v, w, p );
};






















void FluidSimProc::SourceSolver( cdouble dt )
{
	double rate = (double)(rand() % 300 + 1) / 100.f;

	for ( int k = 0; k < 128; k++ ) for ( int j = 0; j < 128; j++ ) for ( int i = 0; i < 128; i++ )
	{
		if ( obs[ix(i,j,k)] < 0.f )
		{
			double pop = -obs[ix(i,j,k)] / 100.f;

			/* add source to grids */
			den[ix(i,j,k)] = DENSITY * rate * dt * pop;

			v[ix(i,j,k)] = VELOCITY * rate * dt * pop;
		}
	}
};

void FluidSimProc::DensitySolver( cdouble dt )
{
	Diffusion( den0, den, DIFFUSION );
	std::swap( den0, den );
	Advection( den, den0, u, v, w, dt );
};

void FluidSimProc::VelocitySolver( cdouble dt )
{
	// diffuse the velocity field (per axis):
	Diffusion( u0, u, VISOCITY );
	Diffusion( v0, v, VISOCITY );
	Diffusion( w0, w, VISOCITY );

	std::swap( u0, u );
	std::swap( v0, v );
	std::swap( w0, w );

	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( u, v, w, div, p );
	
	// advect the velocity field (per axis):
	Advection( u0, u, u, v, w, dt );
	Advection( v0, v, u, v, w, dt );
	Advection( w0, w, u, v, w, dt );

	std::swap( u0, u );
	std::swap( v0, v );
	std::swap( w0, w );
	
	// stabilize it: (vx0, vy0 are whatever, being used as temporaries to store gradient field)
	Projection( u, v, w, div, p );
};