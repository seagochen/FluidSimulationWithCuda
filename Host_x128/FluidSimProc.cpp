/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Apr 03, 2014
* <File Name>     FluidSimProc.cpp
*/

#include <time.h>
#include <iostream>
#include <utility>
#include "MacroDefinition.h"
#include "FluidSimProc.h"
#include "MacroDefinition.h"

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
	u = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	v = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	w = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	u0 = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	v0 = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	w0 = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	den = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	den0 = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	p = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	obs = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );
	div = (double*) malloc ( 128 * 128 * 128 * sizeof(double) );

	visual = (uchar*) malloc ( 128 * 128 * 128 * sizeof(uchar) );

	if ( u eqt nullptr or v eqt nullptr or w eqt nullptr ) goto Error;
	if ( u0 eqt nullptr or v0 eqt nullptr or w0 eqt nullptr ) goto Error;
	if ( den eqt nullptr or den0 eqt nullptr ) goto Error;
	if ( p eqt nullptr or obs eqt nullptr or div eqt nullptr ) goto Error;
	if ( visual eqt nullptr ) goto Error;

	goto Success;

Error:
		cout << "create buffers for host failed" << endl;
		FreeResource();
		exit(1);

Success:
		cout << "all resource created" << endl;
};


void FluidSimProc::FreeResource( void )
{
	SAFE_FREE_PTR( u );
	SAFE_FREE_PTR( v );
	SAFE_FREE_PTR( w );
	SAFE_FREE_PTR( u0 );
	SAFE_FREE_PTR( v0 );
	SAFE_FREE_PTR( w0 );
	SAFE_FREE_PTR( den );
	SAFE_FREE_PTR( den0 );
	SAFE_FREE_PTR( p );
	SAFE_FREE_PTR( obs );
	SAFE_FREE_PTR( div );
	SAFE_FREE_PTR( visual );

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

	fluid->volume.ptrData = visual;
};


void FluidSimProc::ClearBuffers( void )
{
	for ( int k = 0; k < 128; k ++ ) for ( int j = 0; j < 128; j++ ) for ( int i = 0; i < 128; i++ )
	{
		u[ix(i,j,k)] = v[ix(i,j,k)] = w[ix(i,j,k)] = 0.f;
		u0[ix(i,j,k)] = v0[ix(i,j,k)] = w0[ix(i,j,k)] = 0.f;
		den[ix(i,j,k)] = den0[ix(i,j,k)] = 0.f;
		p[ix(i,j,k)] = div[ix(i,j,k)] = obs[ix(i,j,k)] = 0.f;
		
		visual[ix(i,j,k)] = 0;
	}

	cout << "call member function ClearBuffers success" << endl;
}


void FluidSimProc::InitBoundary( void )
{
	cint halfx = 128 / 2;
	cint halfz = 128 / 2;

	for ( int k = 0; k < 128; k ++ ) for ( int j = 0; j < 128; j++ ) for ( int i = 0; i < 128; i++ )
	{
		if ( j < 3 and 
			i >= halfx - 2 and i < halfx + 2 and 
			k >= halfz - 2 and k < halfz + 2 )
			obs[ix(i,j,k)] = MACRO_BOUNDARY_SOURCE;
		else
			obs[ix(i,j,k)] = MACRO_BOUNDARY_BLANK;
	}
	cout << "call member function InitBoundary success" << endl;
};


void FluidSimProc::GenerVolumeImg( void )
{
	for ( int k = 0; k < 128; k ++ ) for ( int j = 0; j < 128; j++ ) for ( int i = 0; i < 128; i++ )
	{
		visual[ix(i,j,k)] = ( den[ix(i,j,k)] > 0.f and den[ix(i,j,k)] < 250.f ) ? 
			(uchar)den[ix(i,j,k)] : 0;
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

	/* duration of adding source */
	t_start = clock();
	SourceSolver( DELTATIME );
	t_finish = clock();
	t_duration = (double)( t_finish - t_start ) / CLOCKS_PER_SEC;
	printf( "%f ", t_duration );

	/* duration of velocity solver */
	t_start = clock();
	VelocitySolver( DELTATIME );
	t_finish = clock();
	t_duration = (double)( t_finish - t_start ) / CLOCKS_PER_SEC;
	printf( "%f ", t_duration );

	/* duration of density solver */
	t_start = clock();
	DensitySolver( DELTATIME );
	t_finish = clock();
	t_duration = (double)( t_finish - t_start ) / CLOCKS_PER_SEC;
	printf( "%f ", t_duration );

	GenerVolumeImg();
	
	RefreshStatus( fluid );

	/* FPS */
	printf( "%d", fluid->fps.uFPS );

	t_totaltimes++;

	printf("\n");
};