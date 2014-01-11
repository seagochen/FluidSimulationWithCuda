/**
* <Author>      Orlando Chen
* <First>       Jan 07, 2014
* <Last>		Jan 08, 2014
* <File>        VolumeRenderingDynamic.h
*/

#ifndef __volume_rendering_dynamic_h_
#define __volume_rendering_dynamic_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <string>

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

	private:
		bool CheckHandleError ( int nShaderObjs, ... );
	};
};

#endif