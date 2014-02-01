/**
* <Author>      Orlando Chen
* <First>       Jan 08, 2014
* <Last>		Feb 01, 2014
* <File>        CUDAFunctionHelper.h
*/

#ifndef __cuda_function_helper_h_
#define __cuda_function_helper_h_

#include <device_launch_parameters.h>
#include "DataStructures.h"

using std::string;

namespace sge
{
	enum SGBOUNDARY
	{
		SG_BD_SOURCE,
		SG_BD_BLANK,
		SG_BD_WALL,
	};
	
	enum SGFIELDTYPE
	{
		SG_DENSITY_FIELD,
		SG_VELOCITY_U_FIELD,
		SG_VELOCITY_V_FIELD,
		SG_VELOCITY_W_FIELD,
	};
	
	enum SGNODECOORD
	{
		SG_CENTER,
		SG_LEFT,
		SG_RIGHT,
		SG_UP,
		SG_DOWN,
		SG_FRONT,
		SG_BACK,
		SG_NO_DEFINE,
	};
	
	enum SGJACOBI
	{
		SG_SOLVE_DENSITY,
		SG_SOLVE_VELOCITY,
	};

	/* standard format of grids */
	typedef struct SGSTDGRID
	{
		SGDOUBLE    u, v, w, dens;
		SGBOUNDARY  obstacle;
	}SGGRID;

	/* structure of host node */
	struct SGHOSTNODE
	{
		SGHOSTNODE *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
		SGBOOLEAN   bActive;
		SGINT3      n3Pos;
		SGSTDGRID  *prtGrids;
	};

	struct SGDEVICEBUFF
	{
		SGSTDGRID *ptrCenter, *ptrLeft, *ptrRight, *ptrUp, *ptrDown, *ptrFront, *ptrBack;
	};
	
	class CUDAFuncHelper
	{
	public:
		SGVOID CheckRuntimeErrors( const char* msg, const char *file, const int line );
		SGVOID DeviceDim2Dx( dim3 *grid_out, dim3 *block_out );
		SGVOID DeviceDim3Dx( dim3 *grid_out, dim3 *block_out );

		SGRUNTIMEMSG CreateDoubleBuffers( SGINT size, SGINT nPtrs, ... );
		SGRUNTIMEMSG CreateIntegerBuffers( SGINT size, SGINT nPtrs, ... );

		void CopyData
			( SGDOUBLE *buffer, const SGDEVICEBUFF *devbuffs, SGFIELDTYPE type, SGNODECOORD coord );
		void CopyData
			( SGDEVICEBUFF *devbuffs, const SGDOUBLE *buffer, SGFIELDTYPE type, SGNODECOORD coord );
	};

	extern void Boundary( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type );
	extern void VelocitySolver( double *u, double *v, double *w, double *div, double *p,
							 double *u0, double *v0, double *w0,
							SGDEVICEBUFF *global, double *stores );
	extern void DensitySolver ( double *u, double *v, double *w, double *dens, double *dens0,
							SGDEVICEBUFF *global, double *stores );
	extern void AddSource( double *buffer, SGSTDGRID *grids, SGFIELDTYPE type );
};

#endif