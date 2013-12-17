#ifndef __main_h_
#define __main_h_

#include <SGE\SGUtils.h>

#include <string>
#include <cstdarg>
#include <iostream>

using namespace sge;
using namespace std;

struct fluidsim
{
	GLuint  hProgram, hBFVert, hBFFrag, hRCVert, hRCFrag;
	GLuint  hCluster;
	GLuint  hTexture1D, hTexture2D, hTexture3D;
	GLuint  hFramebuffer;
	Shader *ptrShader;
	GLint   nAngle, nVolWidth, nVolHeight, nVolDepth;
	GLint   nScrWidth, nScrHeight;
	GLfloat fStepsize;
};


// Initialize 'shader helper' class, and set the 'handles' of OpenGL program and shader objects
extern void CreateShaders ( Shader *shader_out, GLuint *prog_out, GLuint *bfVert_out, GLuint *bfFrag_out, GLuint *rcVert_out, GLuint *rcFrag_out );

// Sets 1-D texture for transfer function
extern GLuint Create1DTransFunc ( void );

// Sets 2-D texture for backface
extern GLuint Create2DBackFace ( GLint width, GLint height );

// Sets 3-D texture for volumetric data 
extern GLuint Create3DVolumetric ( const char *filename, GLint width, GLint height, GLint depth );

// Sets framebuffer
extern GLuint CreateFrameBuffer ( GLuint texObj, GLint width, GLint height );

// Sets rendering face
extern void RenderingFace ( GLenum cullFace, GLfloat angle, GLuint program, GLuint cluster, GLint width, GLint height );

extern void SetVolumeInfoUinforms ( GLuint program, GLuint Tex1DTrans, GLuint Tex2DBF, GLuint Tex3DVol, GLfloat width, GLfloat height, GLfloat stepsize );

extern GLuint InitVerticesBufferObj ( void );

#endif