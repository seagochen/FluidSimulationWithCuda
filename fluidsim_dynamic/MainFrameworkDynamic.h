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
	typedef class Framework_v1_0
	{
	private:
		SGUCHAR  *DefaultTransFunc( SGVOID );
		SGHANDLER Create1DTransFunc( GLubyte *transfer );
		SGHANDLER Create2DCanvas( fluidsim *fluid );
		SGHANDLER Create2DFrameBuffer( fluidsim *fluid );
		SGHANDLER Create3DVolumetric( SGVOID );
		SGHANDLER CreateVerticesBufferObj( SGVOID );

	private:
		SGVOID SetVolumeInfoUinforms( fluidsim *fluid );
		SGVOID RenderingFace( GLenum cullFace, fluidsim *fluid );
		SGVOID CreateShaderProg( fluidsim *fluid );
		SGVOID LoadVolumeSource( SGCONSTCHAR *szRawFile, fluidsim *fluid );

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