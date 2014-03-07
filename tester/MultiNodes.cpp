#include <stdarg.h> 
#include <iostream>
#include <vector>
#include "BasicType.h"

#define HNODES_X 3
#define GNODES_X 2
#define CURSOR_X 2

using std::endl;
using std::cout;
using std::vector;

#define MACRO_FALSE 0
#define MACRO_TRUE  1

struct SimNode
{
	int  updated;
	int3 n3Pos;
	SimNode *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
};

int *host_nodes, *gpu_nodes;
vector<SimNode*> host_link, gpu_link;
int3 m_cursor;

bool CreateHostBuffers( size_t size, int nPtrs, ... )
{
	void **ptr;

	va_list ap;
	va_start( ap, nPtrs );
	for ( int i = 0; i < nPtrs; i++ )
	{
		ptr = va_arg( ap, void** );
		*ptr = (void*)malloc( size );
		
		if ( *ptr eqt nullptr )
		{
			printf( "malloc space failed!\n" );
			return false;
		}
	}
	va_end( ap );

	return true;
};

void MallocSpace()
{
	if ( not CreateHostBuffers(sizeof(int)*HNODES_X*HNODES_X*HNODES_X, 1, &host_nodes ) )
	{
		printf("create array failed\n");
		free(host_nodes);
		exit(1);
	}
	if ( not CreateHostBuffers(sizeof(int)*GNODES_X*GNODES_X*GNODES_X, 1, &gpu_nodes) )
	{
		printf("create array failed\n");
		free(host_nodes);
		free(gpu_nodes);
		exit(1);
	}
};

void CreateLink()
{
	/* create host nodes */
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		/* create a host node */
		SimNode *ptr = (SimNode*)malloc(sizeof(SimNode));
		ptr->ptrFront = ptr->ptrBack = nullptr;
		ptr->ptrLeft  = ptr->ptrRight = nullptr;
		ptr->ptrDown  = ptr->ptrUp = nullptr;
		ptr->updated  = MACRO_FALSE;

		host_link.push_back( ptr );
	}

	/* create gpu nodes */
	for ( int i = 0; i < GNODES_X * GNODES_X * GNODES_X; i++ )
	{
		/* create a host node */
		SimNode *ptr = (SimNode*)malloc(sizeof(SimNode));
		ptr->ptrFront = ptr->ptrBack = nullptr;
		ptr->ptrLeft  = ptr->ptrRight = nullptr;
		ptr->ptrDown  = ptr->ptrUp = nullptr;
		ptr->updated  = MACRO_FALSE;

		gpu_link.push_back( ptr );
	}
};

void CreateTopology()
{
	/* create host link topology */
	for ( int i = 0; i < HNODES_X; i++ )
	{
		for ( int j = 0; j < HNODES_X; j++ )
		{
			for ( int k = 0; k < HNODES_X; k++ )
			{
				/* left */
				if ( i >= 1 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrLeft  = host_link[cudaIndex3D( i-1, j, k, HNODES_X )];
				/* right */
				if ( i <= HNODES_X - 2 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrRight = host_link[cudaIndex3D( i+1, j, k, HNODES_X )];
				/* down */
				if ( j >= 1 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrDown  = host_link[cudaIndex3D( i, j-1, k, HNODES_X )];
				/* up */
				if ( j <= HNODES_X - 2 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrUp    = host_link[cudaIndex3D( i, j+1, k, HNODES_X )];
				/* back */
				if ( k >= 1 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrBack  = host_link[cudaIndex3D( i, j, k-1, HNODES_X )];
				/* front */
				if ( k <= HNODES_X - 2 )
					host_link[cudaIndex3D( i, j, k, HNODES_X )]->ptrFront = host_link[cudaIndex3D( i, j, k+1, HNODES_X )];

				host_link[cudaIndex3D( i, j, k, HNODES_X )]->n3Pos.x = i;
				host_link[cudaIndex3D( i, j, k, HNODES_X )]->n3Pos.y = j;
				host_link[cudaIndex3D( i, j, k, HNODES_X )]->n3Pos.z = k;
			}
		}
	}

	/* create gpu link topology */
	for ( int i = 0; i < GNODES_X; i++ )
	{
		for ( int j = 0; j < GNODES_X; j++ )
		{
			for ( int k = 0; k < GNODES_X; k++ )
			{
				/* left */
				if ( i >= 1 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrLeft  = gpu_link[cudaIndex3D( i-1, j, k, GNODES_X )];
				/* right */
				if ( i <= GNODES_X - 2 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrRight = gpu_link[cudaIndex3D( i+1, j, k, GNODES_X )];
				/* down */
				if ( j >= 1 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrDown  = gpu_link[cudaIndex3D( i, j-1, k, GNODES_X )];
				/* up */
				if ( j <= GNODES_X - 2 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrUp    = gpu_link[cudaIndex3D( i, j+1, k, GNODES_X )];
				/* back */
				if ( k >= 1 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrBack  = gpu_link[cudaIndex3D( i, j, k-1, GNODES_X )];
				/* front */
				if ( k <= GNODES_X - 2 )
					gpu_link[cudaIndex3D( i, j, k, GNODES_X )]->ptrFront = gpu_link[cudaIndex3D( i, j, k+1, GNODES_X )];
			}
		}
	}
};

void InitHostNode()
{
	for ( int i = 0; i < HNODES_X * HNODES_X * HNODES_X; i++ )
	{
		host_nodes[i] = i;
	}
};

void PrintHostNode()
{
	printf( "\n------------\n" );
	for ( int k = 0; k < HNODES_X; k++ )
	{
		for ( int j = 0; j < HNODES_X; j++ )
		{
			for ( int i = 0; i < HNODES_X; i++ )
			{
				printf( "%d ", host_nodes[cudaIndex3D(i,j,k,HNODES_X)] );
			}
			printf( "\n" );
		}
		printf( "------------\n" );
	}
};

void PrintDevNode()
{
	printf( "\n------------\n" );
	for ( int k = 0; k < GNODES_X; k++ )
	{
		for ( int j = 0; j < GNODES_X; j++ )
		{
			for ( int i = 0; i < GNODES_X; i++ )
			{
				printf( "%d ", gpu_nodes[cudaIndex3D(i,j,k,GNODES_X)] );
			}
			printf( "\n" );
		}
		printf( "------------\n" );
	}
}

void PickHostNode()
{
	for ( int k = 0; k < GNODES_X; k++ )
	{
		for ( int j = 0; j < GNODES_X; j++ )
		{
			for ( int i = 0; i < GNODES_X; i++ )
			{
				gpu_nodes[cudaIndex3D(i,j,k,GNODES_X)] =
					host_nodes[cudaIndex3D(i+m_cursor.x,j+m_cursor.y,k+m_cursor.z,HNODES_X)];
			}
		}
	}
};

int main()
{
	MallocSpace();
	CreateLink();
	CreateTopology();
	InitHostNode();

	cout << " host info " << endl;
	PrintHostNode();

	cout << " gpu info " << endl;
	for ( int k = 0; k < CURSOR_X; k++ )
	{
		for ( int j = 0; j < CURSOR_X; j++ )
		{
			for ( int i = 0; i < CURSOR_X; i++ )
			{
				m_cursor.x = i;
				m_cursor.y = j;
				m_cursor.z = k;

				PickHostNode();
				PrintDevNode();
			}
		}
	}

	printf("bye!\n");
	free(host_nodes);
	free(gpu_nodes);
	return 0;
};