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
* <First>       Dec 15, 2013
* <Last>		Dec 21, 2013
* <File>        fluidsim.h
*/

#ifndef __fluid_sim_h_
#define __fluid_sim_h_

#include <GL\glew.h>
#include <GL\freeglut.h>

#include <SGE\SGUtils.h>

struct fluidsim
{
	/// Handles for shader objects, and programs ///
	GLuint hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
	/// Handles for vertices buffer object ///
	GLuint hCluster;
	/// Handles for textures, and frame buffer ///
	GLuint hTexture1D, hTexture2D, hTexture3D, hFramebuffer;
	/// Shader Helper ///
	sge::Shader *ptrShader;
	/// 3-D volumetric data ///
	GLubyte *ptrData;
	/// 3-D volumetric data information ///
	GLint nVolWidth, nVolHeight, nVolDepth;
	/// Source files for GLSL ///
	char *szCanvasVert, *szCanvasFrag, *szVolumVert, *szVolumFrag;
	/// Screen parameters ///	
	GLint nAngle;
	bool  bFullScreen;
	/// Canvas information ///
	GLint nCanvasWidth, nCanvasHeight;
	/// Stride of ray casting ///
	GLfloat  fStepsize;
};


struct param
{
	const static int nGrids_X  = 64;
	const static int nSim_Size = nGrids_X * nGrids_X * nGrids_X;
	const static int nGridGhostHeader = 0;
	const static int nGridGhostTail = nGrids_X - 1;
	const static int nGridSimHeader = 1;
	const static int nGridSimTail   = nGridGhostTail - 1;
};

#endif