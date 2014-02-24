/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <First>       Dec 21, 2013
* <Last>		Dec 21, 2013
* <File>        volumeHelper.h
*/

#ifndef __volume_helper_h_
#define __volume_helper_h_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

#include <string>

#include "fluidsim.h"

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