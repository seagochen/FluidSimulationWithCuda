/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Dec 15, 2013
* <Last Time>     Feb 14, 2014
* <File Name>     FluidSimProc.h
*/


#ifndef __fluid_simulation_process_h_
#define __fluid_simulation_process_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <vector>
#include "CUDAMacroDef.h"

#define Simul_Size          GRIDS_X*GRIDS_X*GRIDS_X

#define DevListNum           11
#define dev_u                dev_list [ 0 ]
#define dev_v                dev_list [ 1 ]
#define dev_w                dev_list [ 2 ]
#define dev_u0               dev_list [ 3 ]
#define dev_v0               dev_list [ 4 ]
#define dev_w0               dev_list [ 5 ]
#define dev_den              dev_list [ 6 ]
#define dev_den0             dev_list [ 7 ]
#define dev_div              dev_list [ 8 ]
#define dev_p                dev_list [ 9 ]
#define dev_grid             dev_list [ 10 ]

#define HostListNum          6
#define host_u               host_list [ 0 ]
#define host_v               host_list [ 1 ]
#define host_w               host_list [ 2 ]
#define host_den             host_list [ 3 ]
#define host_div             host_list [ 4 ]
#define host_p               host_list [ 5 ]


namespace sge
{
	class FluidSimProc
	{
	private:
		std::vector <double*> dev_list;
		std::vector <double*> host_list;
		GLubyte *host_data;
		unsigned char *dev_data;

	public:
		FluidSimProc ( FLUIDSPARAM *fluid );

		void FluidSimSolver ( FLUIDSPARAM *fluid );
		void FreeResource ( void );
		void ZeroBuffers ( void );

	private:
		SGRUNTIMEMSG AllocateResourcePtrs ( FLUIDSPARAM *fluid );
		void DensitySolver ( void );
		void VelocitySolver ( void );
		void PickData ( FLUIDSPARAM *fluid );
		void CopyDataToHost ( void );
		void CopyDataToDevice ( void );
	};
};

#endif