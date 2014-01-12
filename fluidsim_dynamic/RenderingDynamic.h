/**
* <Author>      Orlando Chen
* <First>       Nov 21, 2013
* <Last>		Jan 12, 2014
* <File>        RenderingDynamic.h
*/

#ifndef __rendering_dynamic_h_
#define __rendering_dynamic_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <string>

#include "FluidSimAreaDynamic.h"

namespace sge
{
	class VolumeHelper
	{
	public:
		GLubyte *DefaultTransFunc ( void );

		GLuint Create1DTransFunc ( GLubyte *transfer );

		GLuint Create2DCanvas ( fluidsim *fluid );

		GLuint Create2DFrameBuffer ( fluidsim *fluid );

		GLuint Create3DVolumetric ( void );

	public:
		GLuint InitVerticesBufferObj ( void );

		void SetVolumeInfoUinforms ( fluidsim *fluid );

		void RenderingFace ( GLenum cullFace, fluidsim *fluid );

		void CreateShaderProg ( fluidsim *fluid );

		void LoadVolumeSource ( const char *szRawFile, fluidsim *fluid );

	private:
		bool CheckHandleError ( int nShaderObjs, ... );
	};
};

#endif