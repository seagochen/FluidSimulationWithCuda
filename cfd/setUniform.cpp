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
* <First>       Dec 12, 2013
* <Last>		Dec 12, 2013
* <File>        setUniform.cpp
*/

#ifndef __set_uniform_cpp_
#define __set_uniform_cpp_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLT/glm/glm.hpp>
#include <GLT/glm/gtc/matrix_transform.hpp>
#include <GLT/glm/gtx/transform2.hpp>
#include <GLT/glm/gtc/type_ptr.hpp>

#include "cfdHeader.h"
#include "GLSLHelper.h"

using namespace std;


/*
-----------------------------------------------------------------------------------------------------------
* @function SetVolumeInfoUinforms
* @author   Orlando Chen
* @date     Dec 5, 2013
* @input    NULL
* @return   NULL
* @bref     When this function is called from external, several uniform information will be set,
* --------- such as screent size, step length, transfer function.
* --------- The entrance points are the front face of agent box, and exit points are the back face.
-----------------------------------------------------------------------------------------------------------
*/
void SetVolumeInfoUinforms ( void )
{
	// Set the uniform of screen size
    GLint screenSizeLoc = glGetUniformLocation ( m_programHandle, "ScreenSize" );
    if ( screenSizeLoc >= 0 )
    {
		// Incoming two value, width and height
		glUniform2f ( screenSizeLoc, (float)m_width, (float)m_height );
    }
    else
    {
		cout << "ScreenSize is not bind to the uniform" << endl;
    }

	// Set the step length
    GLint stepSizeLoc = glGetUniformLocation ( m_programHandle, "StepSize" );
	if ( stepSizeLoc >= 0 )
    {
		// Incoming one value, the step size
		glUniform1f ( stepSizeLoc, StepSize );
    }
    else
    {
		cout << "StepSize is not bind to the uniform" << endl;
    }
    
	// Set the transfer function
	GLint transferFuncLoc = glGetUniformLocation ( m_programHandle, "TransferFunc" );
    if ( transferFuncLoc >= 0 )
	{
		/**
		* void glActiveTexture ( GLenum texture );
		* 
		* glActiveTexture selects which texture unit subsequent texture state calls will affect.
		* The number of texture units an implementation supports is implementation dependent, but must be at least 80.
		*/
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_1D, m_tffTexObj );
		glUniform1i ( transferFuncLoc, 0 );
    }
    else
    {
		cout << "TransferFunc is not bind to the uniform" << endl;
    }

	// Set the back face as exit point for ray casting
	GLint backFaceLoc = glGetUniformLocation ( m_programHandle, "ExitPoints" );
	if ( backFaceLoc >= 0 )
    {
		glActiveTexture ( GL_TEXTURE1 );
		glBindTexture(GL_TEXTURE_2D, m_bfTexObj);
		glUniform1i(backFaceLoc, 1);
    }
    else
    {
		cout << "ExitPoints is not bind to the uniform" << endl;
    }

	// Set the uniform to hold the data of volumetric data
	GLint volumeLoc = glGetUniformLocation(m_programHandle, "VolumeTex");
	if (volumeLoc >= 0)
    {
		glActiveTexture ( GL_TEXTURE2 );
		glBindTexture(GL_TEXTURE_3D, m_volTexObj);
		glTexImage3D (GL_TEXTURE_3D, 0, GL_INTENSITY, Grids_X, Grids_X, Grids_X, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, host_den);
		glUniform1i(volumeLoc, 2);
    }
    else
    {
		cout << "VolumeTex is not bind to the uniform" << endl;
    }    
};


/*
-----------------------------------------------------------------------------------------------------------
* @function RenderingFace
* @author   Orlando Chen
* @date     Dec 5, 2013
* @input    GLenum cullFace
* @return   NULL
* @bref     Under the process of OnDisplay, this function will be called twice. During The first calling, 
* --------- the backface of boundbox was rendered, and the second time, frontface was rendered. The data
* --------- of both faces will be keeped, and used as entry points for ray casting in and out the box.
-----------------------------------------------------------------------------------------------------------
*/
void RenderingFace ( GLenum cullFace )
{
	using namespace glm;

	// Check OpenGL runtime error
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );

	// Clear background color and depth buffer
    glClearColor ( 0.2f,0.2f,0.2f,1.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
	//  Set projection and lookat matrix
    mat4 projection = perspective ( 60.0f, (GLfloat)m_width/m_height, 0.1f, 400.f );
    mat4 view = lookAt (
		vec3(0.0f, 0.0f, 2.0f),
		vec3(0.0f, 0.0f, 0.0f), 
		vec3(0.0f, 1.0f, 0.0f));

	// Set model view matrix
    mat4 model = mat4(1.0f);
	model = model * rotate ( (float)m_angle, vec3(0.0f, 1.0f, 0.0f) );
    
	// Rotate and translate the view matrix, let object seems to "stand up"
	// Because, original volumetric data is "lying down" on ground.
	model = model * rotate ( 90.0f, vec3(1.0f, 0.0f, 0.0f) );
	model = model * translate ( vec3(-0.5f, -0.5f, -0.5f) ); 
    
	// Finally, we focus on setting the Model View Projection Matrix (MVP matrix)
	// Notice that the matrix multiplication order: reverse order of transform
    mat4 mvp = projection * view * model;

	// Returns an integer that represents the location of a specific uniform variable within a shader program
    GLuint mvpIdx = glGetUniformLocation ( m_programHandle, "MVP" );
    
	if ( mvpIdx >= 0 )
    {
    	glUniformMatrix4fv ( mvpIdx, 1, GL_FALSE, &mvp[0][0] );
    }
    else
    {
    	cerr << "can't get the MVP" << endl;
    }

	// Check OpenGL runtime error
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );
    
	// Draw agent box
	glEnable ( GL_CULL_FACE );
	glCullFace ( cullFace );
	glBindVertexArray ( m_cluster );
	glDrawElements ( GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint *)NULL );
	glDisable ( GL_CULL_FACE );

	// Check OpenGL runtime error
    helperCheckOpenGLStatus ( __FILE__, __LINE__ );
}

#endif