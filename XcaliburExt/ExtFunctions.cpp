/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Mar 23, 2014
* <Last Time>     Mar 25, 2014
* <File Name>     ExtFunctions.cpp
*/


#include "MacroDefinition.h"
#include "ExtFunctions.h"
#include "Kernels.h"

using namespace sge;

bool CreateCompNodesForDevice
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, 
	vector<double*> *vectVelW, vector<double*> *vectObst, 
	FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};

bool CreateCompNodesForDevice( vector<double*> *vectBuf, FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateDeviceBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateDeviceBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectBuf->push_back( ptrD );
	}

	return true;
};

bool CreateCompNodesForHost
	( vector<double*> *vectDens, vector<double*> *vectVelU, vector<double*> *vectVelV, vector<double*> *vectVelW,
	vector<double*> *vectObst, FunctionHelper *helper, size_t size, size_t nodes )
{
	for ( int i = 0; i < nodes; i++ )
	{
		double *ptrD, *ptrU, *ptrV, *ptrW, *ptrO;

		if ( helper->CreateHostBuffers( size, 1, &ptrD ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrO ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrU ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrV ) not_eq SG_RUNTIME_OK ) return false;
		if ( helper->CreateHostBuffers( size, 1, &ptrW ) not_eq SG_RUNTIME_OK ) return false;

		vectDens->push_back( ptrD );
		vectVelU->push_back( ptrU );
		vectVelV->push_back( ptrV );
		vectVelW->push_back( ptrW );
		vectObst->push_back( ptrO );
	}

	return true;
};