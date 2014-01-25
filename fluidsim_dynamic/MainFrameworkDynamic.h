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

namespace sge
{
	typedef GLenum GLENUM;

	typedef class Framework_v1_0
	{
	private:
		SGUCHAR  *DefaultTransFunc( SGVOID );
		SGHANDLER Create1DTransFunc( GLubyte *transfer );
		SGHANDLER Create2DCanvas( FLUIDSPARAM *fluid );
		SGHANDLER Create2DFrameBuffer( FLUIDSPARAM *fluid );
		SGHANDLER Create3DVolumetric( SGVOID );
		SGHANDLER CreateVerticesBufferObj( SGVOID );

	private:
		SGVOID SetVolumeInfoUinforms( FLUIDSPARAM *fluid );
		SGVOID RenderingFace( GLENUM cullFace, FLUIDSPARAM *fluid );
		SGVOID CreateShaderProg( FLUIDSPARAM *fluid );
		SGVOID LoadVolumeSource( SGCONSTCHAR *szRawFile, FLUIDSPARAM *fluid );

	private:
		SGBOOLEAN CheckHandleError( SGINT nShaderObjs, ... );

	public:
		Framework_v1_0( SGVOID );

	public:
		SGVOID onCreate( SGVOID );
		SGVOID onDestroy( SGVOID );
		SGVOID onDisplay( SGVOID );
		SGVOID onReshape( SGUINT width, SGUINT height );
		SGVOID onIdle( SGVOID );
		SGVOID onKeyboard( SGKEYS keys, SGKEYSTATUS status );
		SGVOID onMouse( SGMOUSE mouse, SGUINT x, SGUINT y, SGINT degree );

	}FrameworkDynamic;
};

#endif