/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Jan 25, 2014
* <File>        MainFrameworkDynamic.h
*/

#ifndef __main_framework_dynamic_h_
#define __main_framework_dynamic_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include "DataStructures.h"
#include "FunctionHelper.h"

namespace sge
{
	typedef GLenum GLENUM;

	typedef class Framework_v1_0
	{
	private:
		static SGUCHAR  *DefaultTransFunc( SGVOID );
		static SGHANDLER Create1DTransFunc( GLubyte *transfer );
		static SGHANDLER Create2DCanvas( FLUIDSPARAM *fluid );
		static SGHANDLER Create2DFrameBuffer( FLUIDSPARAM *fluid );
		static SGHANDLER Create3DVolumetric( SGVOID );
		static SGHANDLER CreateVerticesBufferObj( SGVOID );

	private:
		static SGVOID SetVolumeInfoUinforms( FLUIDSPARAM *fluid );
		static SGVOID RenderingFace( GLENUM cullFace, FLUIDSPARAM *fluid );
		static SGVOID CreateShaderProg( FLUIDSPARAM *fluid );
		static SGVOID LoadVolumeSource( SGCONSTCHAR *szRawFile, FLUIDSPARAM *fluid );

	private:
		static SGBOOLEAN CheckHandleError( SGINT nShaderObjs, ... );
		static SGVOID CountFPS();
		static DWORD WINAPI FluidSimulationProc ( LPVOID lpParam );

	public:
		Framework_v1_0( SGMAINACTIVITY **activity, SGUINT width, SGUINT height );

	public:
		static SGVOID onMouse( SGMOUSE mouse, unsigned x, unsigned y, int degree );
		static SGVOID onKeyboard( SGKEYS keys, SGKEYSTATUS status );
		static SGVOID onDestroy( SGVOID );
		static SGVOID onDisplay( SGVOID );		
		static SGVOID onCreate( SGVOID );
	}FrameworkDynamic;
};

#endif