#ifndef __main_h_
#define __main_h_

#include <GL\glew32.h>
#include <GL\freeglut.h>

#include <SGE\SGUtils.h>

#include <string>
#include <cstdarg>
#include <iostream>

using namespace sge;
using namespace std;

#define BACKFACE_SIZE_X  400


// Initialize 'shader helper' class, and set the 'handles' of OpenGL program and shader objects
extern void CreateShaders 
	( Shader *shader_out, GLuint *prog_out, 
	GLuint *bfVert_out, GLuint *bfFrag_out, GLuint *rcVert_out, GLuint *rcFrag_out );

// Sets 1-D texture for transfer function
extern GLuint Create1DTransFunc ( void );

// Sets 2-D texture for backface
extern GLuint Create2DBackFace ( void );

// Sets 3-D texture for volumetric data 
extern GLuint Create3DVolumetric ( void );

// Sets framebuffer
extern GLuint CreateFrameBuffer ( GLuint texObj );

// Sets rendering face
extern void RenderingFace ( GLenum cullFace, GLfloat angle, GLuint program, GLuint cluster );

extern void SetVolumeInfoUinforms ( GLuint program, GLuint Tex1DTrans, GLuint Tex2DBF, GLuint Tex3DVol, 
							GLfloat width, GLfloat height, GLfloat stepsize );

extern GLuint InitVerticesBufferObj ( void );

#endif