#include "FunctionHelper.h"
//#include "Kernels.h"
#include <iostream>
#include <vector>

typedef const int cint;
typedef const double cdouble;

using namespace std;
using namespace sge;

int ix( cint i, cint j, cint k, cint tilex, cint tiley, cint tilez )
{
	if ( i < 0 or i >= tilex ) return -1;
	if ( j < 0 or j >= tiley ) return -1;
	if ( k < 0 or k >= tilez ) return -1;

	return i + j * tilex + k * tilex * tiley;
};

inline int _round( double x)
{
     return (x >= 0) ? (int)(x + 0.5) : (int)(x - 0.5);
};

void kernelAssembleCompBufs( int *dst, cint dstx, cint dsty, cint dstz, 
									  int *src, cint srcx, cint srcy, cint srcz,
									  cint offi, cint offj, cint offk, 
									  cdouble zoomx, cdouble zoomy, cdouble zoomz )
{
	for ( int k = 0; k < srcz; k++ ) for ( int j = 0; j < srcy; j++ ) for ( int i = 0; i < srcx; i++ )
	{
		int dsti, dstj, dstk;

#if 1
		dsti = offi * srcx + i;
		dstj = offj * srcy + j;
		dstk = offk * srcz + k;
#else
		dsti = _round( (offi * srcx + i) * zoomx );
		dstj = _round( (offj * srcy + j) * zoomy );
		dstk = _round( (offk * srcz + k) * zoomz );
#endif

		if ( dsti < 0 ) dsti = 0;
		if ( dstj < 0 ) dstj = 0;
		if ( dstk < 0 ) dstk = 0;
		if ( dsti >= dstx ) dsti = dstx - 1;
		if ( dstj >= dsty ) dstj = dsty - 1;
		if ( dstk >= dstz ) dstk = dstz - 1;

		dst[ix(dsti, dstj, dstk, dstx, dsty, dstz)] = src[ix(i, j, k, srcx, srcy, srcz)];
	}
}


void kernelDeassembleCompBufs( int *dst, cint dstx, cint dsty, cint dstz, 
										 cint *src, cint srcx, cint srcy, cint srcz,
										 cint offi, cint offj, cint offk, 
										 cdouble zoomx, cdouble zoomy, cdouble zoomz )
{
	for ( int k = 0; k < srcz; k++ ) for ( int j = 0; j < srcy; j++ ) for ( int i = 0; i < srcx; i++ )
	{
#if 0
	double srci, srcj, srck;

	srci = ( i + offi * dstx ) * zoomx;
	srcj = ( j + offj * dsty ) * zoomy;
	srck = ( k + offk * dstz ) * zoomz;

	if ( srci < 0 ) srci = 0.f;
	if ( srcj < 0 ) srcj = 0.f;
	if ( srck < 0 ) srck = 0.f;
	if ( srci >= srcx ) srci = srcx - 1.f;
	if ( srcj >= srcy ) srcj = srcy - 1.f;
	if ( srck >= srcz ) srck = srcz - 1.f;

	dst[ix(i, j, k, dstx, dsty, dstz)] = atomicTrilinear( src, srci, srcj, srck, srcx, srcy, srcz );
#else
	int srci, srcj, srck;

	srci = i + offi * dstx;
	srcj = j + offj * dsty;
	srck = k + offk * dstz;

	if ( srci < 0 ) srci = 0;
	if ( srcj < 0 ) srcj = 0;
	if ( srck < 0 ) srck = 0;
	if ( srci >= srcx ) srci = srcx - 1;
	if ( srcj >= srcy ) srcj = srcy - 1;
	if ( srck >= srcz ) srck = dstz - 1;

	dst[ix(i, j, k, dstx, dsty, dstz)] = src[ix(srci, srcj, srck, srcx, srcy, srcz)];
#endif
	}
};

#if 0

void main()
{
	int *ptrGrids = (int*) calloc ( 20 * 20 * 20, sizeof(int) );
	vector<int*> list;

	FunctionHelper helper;

	helper.CreateCompNodesForHost( &list, 10 * 10 * 10 * sizeof(int), 2 * 2 * 2 );

	printf( "size of list: %d\n", list.size() );

	
	for ( int n = 0; n < 8; n++ )
	for ( int k = 0; k < 10; k++ )
	{
		for ( int j = 0; j < 10; j++ )
		{
			for ( int i = 0; i < 10; i++ )
			{
				list[n][ix(i,j,k,10,10,10)] = n;
			}
		}
	}

	printf( "passed here 1\n" );

	for ( int k = 0; k < 2; k++ )
	{
		for ( int j = 0; j < 2; j++ )
		{
			for ( int i = 0; i < 2; i++ )
			{
				kernelAssembleCompBufs( 
					ptrGrids, 20, 20, 20,
					list[ix(i,j,k,2,2,2)], 10, 10, 10,
					i, j, k,
					1.f, 1.f, 1.f );
			}
		}
	}

	printf( "passed here 2\n" );

#if 0
	for ( int j = 0; j < 20; j++ )
	{
		for ( int k = 0; k < 20; k++ )
		{
			for ( int i = 0; i < 20; i++ )
			{
				printf( "%d ", ptrGrids[ix(i,j,k,20,20,20)] );
			}
			printf( "\n" );
		}
		printf( "\n" );
	}
#endif

	printf( "passed here 3\n" );
	
	for ( int k = 0; k < 2; k++ )
	{
		for ( int j = 0; j < 2; j++ )
		{
			for ( int i = 0; i < 2; i++ )
			{
				kernelDeassembleCompBufs(
					list[ix(i,j,k,2,2,2)], 10, 10, 10,
					ptrGrids, 20, 20, 20,					
					i, j, k,
					1.f, 1.f, 1.f );
			}
		}
	}

	for ( int j = 0; j < 10; j++ )
	{
		for ( int k = 0; k < 10; k++ )
		{
			for ( int i = 0; i < 10; i++ )
			{
				printf( "%d ", list[5][ix(i,j,k,10,10,10)] );
			}
			printf( "\n" );
		}
		printf( "\n" );
	}

};

#endif

#if 0

//void kernelCompressGrid( double *dst, cdouble *src, )
//{};

void Shell3( cint *cell, cint length )
{
	for ( int i = 0; i < length; i++ )
		printf( "%d ", cell[i] );
}

void Shell2( cint *cell, cint length )
{
	Shell3( cell, length );
}

void Shell1( cint *cell, cint length )
{
	Shell2( cell, length );
}

void main()
{
	int *ptr = (int*)calloc( 10, sizeof(int) );

	for ( int i = 0; i < 10; i++ )
		ptr[i] = i;

	Shell1( ptr, 10 );

	system( "pause" );
}

#endif

void kernelCompressGrids( double *dst, cdouble *src, 
						 cint srcx, cint srcy, cint srcz,
						 cint dstx, cint dsty, cint dstz,
						 cint zoomx, cint zoomy, cint zoomz )
{
	if ( dstx * zoomx > srcx ) { cout << "skip x" << endl; return; }
	if ( dsty * zoomy > srcy ) { cout << "skip y" << endl; return; }
	if ( dstz * zoomz > srcz ) { cout << "skip z" << endl; return; }

	for ( int k = 0; k < dstz; k++ ) for ( int j = 0; j < dsty; j++ ) for ( int i = 0; i < dstx; i++ )
	{
		double sum = 0.f;

		for ( int zk = 0; zk < zoomz; zk++ ) for ( int zj = 0; zj < zoomy; zj++ ) for ( int zi = 0; zi < zoomx; zi++ )
		{
			sum += src[ix(i*zoomx+zi, j*zoomy+zj, k*zoomz+zk, srcx, srcy, srcz)];
		}
		dst[ix(i,j,k,dstx,dsty,dstz)] = sum / ( zoomx * zoomy * zoomz );
	}
};

void main()
{
	double *src = (double*)malloc( sizeof(double) * 20 * 20 * 20 );
	double *dst = (double*)malloc( sizeof(double) * 10 * 10 * 10 );

	for ( int i = 0; i < 20 * 20 * 20; i++ ) src[i] = i;

	kernelCompressGrids( dst, src, 
		20, 20, 20,
		10, 10, 10,
		2, 2, 2 );

	for ( int k = 0; k < 10; k++ )
	{
		for ( int j = 0; j < 10; j++ )
		{
			for ( int i = 0; i < 10; i++ )
			{
				printf( "%d ", (int)dst[ix(i,j,k,10,10,10)] );
			}
			printf( "\n" );
		}
		printf( "\n" );
	}
};