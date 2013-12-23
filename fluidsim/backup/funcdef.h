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
* <Last>		Dec 17, 2013
* <File>        funcdef.h
*/

#ifndef __function_define_h_
#define __function_define_h_

#include <SGE\SGUtils.h>

#include <string>
#include <cstdarg>
#include <iostream>

using namespace sge;
using namespace std;

struct fluidsim
{
	GLuint   hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
	GLuint   hCluster;
	GLuint   hTexture1D, hTexture2D, hTexture3D;
	GLuint   hFramebuffer;
	Shader  *ptrShader;
	GLubyte *ptrData;
	GLint    nAngle, nVolWidth, nVolHeight, nVolDepth;
	GLint    nScrWidth, nScrHeight;
	GLint    nCanvasWidth, nCanvasHeight;
	GLfloat  fStepsize;
};


#pragma region functions used in dynamics volumetric rendering
// Initialize 'shader helper' class, and set the 'handles' of OpenGL program and shader objects
extern void CreateShaders ( fluidsim *fluid );

// Sets 1-D texture for transfer function
extern GLuint Create1DTransFunc ( void );

// Sets 2-D texture for backface
extern GLuint Create2DBackFace ( fluidsim *fluid );

// Sets 3-D texture for volumetric data 
extern GLuint Create3DVolumetric ( void );

// Sets framebuffer
extern GLuint CreateFrameBuffer ( fluidsim *fluid );

// Sets rendering face
extern void RenderingFace ( GLenum cullFace, fluidsim *fluid );

// Sets uniform parameters
extern void SetVolumeInfoUinforms ( fluidsim *fluid );

// Create vertices buffer object
extern GLuint InitVerticesBufferObj ( void );  
#pragma endregion


#pragma region functions used in fluid simulation
// Initialize the stage of fluid simulation, yield the velocity and density
extern void InitSimGrid ( float *gridDensity, float *gridU, float *gridV, float *gridW );

// Release the resource of stack
extern void FreeResourcePtrs ( void );

// Zero the data that used in fluid simulation
extern void ZeroData ( void );

// Allocate (instanced) the resource
extern SGRUNTIMEMSG AllocateResourcePtrs ( void );

// Solving the density by Euler method
extern void DensitySolver ( float *dens, float *dens0, float *u, float *v, float *w );

// Solving the velocity by Euler method
extern void VelocitySolver ( float *u, float *v, float *w, float *u0, float *v0, float *w0 );

// Interpolate the dataset
extern void DensityItp ( float *dens, GLubyte *data );
#pragma endregion


#endif