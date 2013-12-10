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
* <First>       Dec 4, 2013
* <Last>		Dec 4, 2013
* <File>        GLSLHelper.h
*/

#ifndef __glsl_helper_h_
#define __glsl_helper_h_

#include <GL\freeglut.h>

/*
-----------------------------------------------------------------------------------------------------------
* @function helperCheckOpenGLStatus
* @author   Orlando Chen
* @date     Dec 4, 2013
* @input    const char *szFile, const int iLine
* @return   NULL
* @bref     Check OpenGL status, if routine occurs runtime exception or error, print out the message
-----------------------------------------------------------------------------------------------------------
*/
void helperCheckOpenGLStatus ( const char *szFile, const int iLine );

/*
-----------------------------------------------------------------------------------------------------------
* @function helperCheckOpenGLStatus
* @author   Orlando Chen
* @date     Dec 4, 2013
* @input    GLuint *hCluster
* @return   NULL
* @bref     Initialize and generate the vertices buffer object
-----------------------------------------------------------------------------------------------------------
*/
void helperInitVerticesBufferObj ( GLuint *hCluster );


/*
-----------------------------------------------------------------------------------------------------------
* @function helperInitPairShadersObj
* @author   Orlando Chen
* @date     Dec 4, 2013
* @input    GLuint *hVShader, const char *szVertSource, GLuint *hFShader, const char *szFragSource
* @return   NULL
* @bref     Load pair of source files and compile the shaders
-----------------------------------------------------------------------------------------------------------
*/
void helperInitPairShadersObj ( GLuint *hVShader, const char *szVertSource, GLuint *hFShader, const char *szFragSource );


/*
-----------------------------------------------------------------------------------------------------------
* @function helperLinkShader
* @author   Orlando Chen
* @date     Dec 4, 2013
* @input    GLuint hShaderProgram, GLuint hVShader, GLuint hFShader
* @return   NULL
* @bref     Link shader
-----------------------------------------------------------------------------------------------------------
*/
void helperLinkShader ( GLuint hShaderProgram, GLuint hVShader, GLuint hFShader );


/*
-----------------------------------------------------------------------------------------------------------
* @function helperCreateShaderProgram
* @author   Orlando Chen
* @date     Dec 4, 2013
* @input    GLuint *hShaderProgram
* @return   NULL
* @bref     Link shader
-----------------------------------------------------------------------------------------------------------
*/
void helperCreateShaderProgram ( GLuint *hShaderProgram );

#endif