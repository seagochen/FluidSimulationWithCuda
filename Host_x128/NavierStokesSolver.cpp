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


void FluidSimProc::Jacobi(double *out, cdouble *in, cdouble diff, cdouble divisor)
{
	double dix = ( divisor > 0 ) ? divisor : 1.f;

    for(int n=0; n<10; n++)
    {
		for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
		{
			out[ix(i, j, k)] = (
				in[ix(i, j, k)] + diff * (
				out[ix(i-1, j, k)] + out[ix(i+1, j, k)] + 
				out[ix(i, j-1, k)] + out[ix(i, j+1, k)] + 
				out[ix(i, j, k-1)] + out[ix(i, j, k+1)] 	)) / dix;
		}
    }
}


#if 0

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
#endif

void FluidSimProc::Diffusion( double *out, cdouble *in, cdouble diff )
{
    double alpha = DELTATIME * diff * 128 * 128 * 128;

    Jacobi( out, in, alpha, 1 + 6 * alpha );
}


void kernelGradient( double *div, double *prs, cdouble *u, cdouble *v, cdouble *w )
{
	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
		cdouble hx = 1.f / (double)128;
		cdouble hy = 1.f / (double)128;
		cdouble hz = 1.f / (double)128;

		// previous instantaneous magnitude of velocity gradient 
		//		= (sum of velocity gradients per axis)/2N:
//		div[ IX(i,j,k,128,128) ] = -0.5f * (
//			hx * ( u[ IX(i+1,j,k,128,128) ] - u[ IX(i-1,j,k,128,128) ] ) +
//			hy * ( v[ IX(i,j+1,k,128,128) ] - v[ IX(i,j-1,k,128,128) ] ) +
//			hz * ( w[ IX(i,j,k+1,128,128) ] - w[ IX(i,j,k-1,128,128) ] ) );
		
		div[ IX(i,j,k,128,128) ] = (double) ( -1.f / 3.f * ( 
			( u[ IX(i+1,j,k,128,128) ] - u[ IX(i-1,j,k,128,128) ] ) / 128.f + 
			( v[ IX(i,j+1,k,128,128) ] - v[ IX(i,j-1,k,128,128) ] ) / 128.f + 
			( w[ IX(i,j,k+1,128,128) ] - w[ IX(i,j,k-1,128,128) ] ) / 128.f ));

		// zero out the present velocity gradient
		prs[ IX(i,j,k,128,128) ] = 0.f;
	}
};


void kernelSubtract( double *u, double *v, double *w, double *prs )
{
	for ( int k = 1; k < 127; k++ ) for ( int j = 1; j < 127; j++ ) for ( int i = 1; i < 127; i++ )
	{
//		u[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i+1,j,k,128,128) ] - prs[ IX(i-1,j,k,128,128) ] );
//		v[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i,j+1,k,128,128) ] - prs[ IX(i,j-1,k,128,128) ] );
//		w[ IX(i,j,k,128,128) ] -= 0.5f * 128 * ( prs[ IX(i,j,k+1,128,128) ] - prs[ IX(i,j,k-1,128,128) ] );

        u[ IX(i,j,k,128,128) ] -= 0.5f * 128.f * ( prs[ IX(i+1,j,k,128,128) ] - prs[ IX(i-1,j,k,128,128) ] );
        v[ IX(i,j,k,128,128) ] -= 0.5f * 128.f * ( prs[ IX(i,j+1,k,128,128) ] - prs[ IX(i,j-1,k,128,128) ] );
        w[ IX(i,j,k,128,128) ] -= 0.5f * 128.f * ( prs[ IX(i,j,k+1,128,128) ] - prs[ IX(i,j,k-1,128,128) ] );

	} 
};

void FluidSimProc::Projection( double *u, double *v, double *w, double *div, double *p )
{
	// the velocity gradient
	kernelGradient( div, p, u, v, w );

	// reuse the Gauss-Seidel relaxation solver to safely diffuse the velocity gradients from p to div
	Jacobi( p, div, 1.f, 6.f );

	// now subtract this gradient from our current velocity field
	kernelSubtract ( u, v, w, p );
};

static int times = 0;

void FluidSimProc::SourceSolver( cdouble dt )
{
	double rate = (double)(rand() % 300 + 1) / 100.f;

	for ( int k = 0; k < 128; k++ ) for ( int j = 0; j < 128; j++ ) for ( int i = 0; i < 128; i++ )
	{
		if ( obs[ix(i,j,k)] < 0.f )
		{
//			double pop = -obs[ix(i,j,k)] / 100.f;

			/* add source to grids */
//			if ( times < 20 )
//			{
//				//den[ix(i,j,k)] = DENSITY * rate * dt * pop;
//				
//				times++;
//			}

			if ( times < 10 )
			//v[ix(i,j,k)] = VELOCITY * rate * dt * pop;
			{
				den[ ix(i,j,k) ] = DENSITY * dt;
				times++;
			}

			v[ix(i,j,k)] = VELOCITY * dt;
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