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
* <First>       Sep 13, 2013
* <Last>		Dec 12, 2013
* <File>        visual_framework.cpp
*/

#ifndef __visual_framework_cpp_
#define __visual_framework_cpp_

#define _CRT_SECURE_NO_WARNINGS

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
using namespace sge;

/*
-----------------------------------------------------------------------------------------------------------
* @function Visual
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    width, height, hActivity
* @return   NULL
* @bref     Constructor
-----------------------------------------------------------------------------------------------------------
*/
Visual::Visual ( GLuint width, GLuint height, MainActivity **hActivity )
{
	// Materialized , and the assignment
	m_mouse    = new _mouse;
	m_fps      = new _fps;
	m_view     = new _viewMatrix;
//	m_font     = new FreeType;
	*hActivity = new MainActivity ( width, height, false );

	m_width    = width;
	m_height   = height;
	m_density  = false;
	
	// etc.
	if ( AllocateResourcePtrs ( ) != SG_RUNTIME_OK )
	{
		FreeResourcePtrs ( );
	}
	else
	{
		ZeroData ( );
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function ~Visual
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Desconstructor   
-----------------------------------------------------------------------------------------------------------
*/
Visual::~Visual ( void )
{
	OnDestroy();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnCreate
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Callback function, firstly to be called when the visual is created, 
*           used to initialize the routine
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnCreate ( void )
{
	// Initialize
	InitViewMatrix ( );
	InitFont ( );
	InitFPS ( );
	InitMouseStatus ( );

	// Call for OpenGL envrionment setup
	Setup();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnResize
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    width, height
* @return   NULL
* @bref     Callback function, to be called when client's window has be changed      
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnResize ( GLuint width, GLuint height )
{
//	m_width  = width;
//	m_height = height;

//	if (height == 0) height = 1;

//	glViewport(0, 0, width, height);

//	glMatrixMode ( GL_PROJECTION );
//	glLoadIdentity ( );
//	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
//	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
//	m_width  = width;
//	m_height = height;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnIdle
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Callback function, to be called when no message has be transfered to SGE      
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnIdle ( void )
{
	VelocitySolver ( host_u, host_v, host_w, host_u0, host_v0, host_w0 );
	DensitySolver ( host_den, host_den0, host_u, host_v, host_w );
	m_angle = (m_angle + 1) % 360;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnDisplay
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Callback function, to render the client      
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnDisplay ( void )
{
	glClear ( GL_COLOR_BUFFER_BIT );

	glEnable ( GL_DEPTH_TEST );
	helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error
	
	// Bind index 0 to the shader input variable "VerPos"
	glBindAttribLocation ( m_programHandle, 0, "VerPos" );
	// Bind index 1 to the shader input variable "VerClr"
	glBindAttribLocation ( m_programHandle, 1, "VerClr" );

    /// Do Render Now!
    glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, m_frameBuffer ); // Chose which framebuffer to render
	glViewport ( 0, 0, m_width, m_height );
    helperLinkShader ( m_programHandle, m_bfVertHandle, m_bfFragHandle ); // Linking shader...
	helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error
    glUseProgram ( m_programHandle ); // Use the shader for rendering the texture
	RenderingFace ( GL_FRONT );   // From front face
    glUseProgram ( 0 );    // Release the shader
	helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error


    glBindFramebuffer ( GL_FRAMEBUFFER, 0 ); // To break the binding
    glViewport ( 0, 0, m_width, m_height );
    helperLinkShader ( m_programHandle, m_rcVertHandle, m_rcFragHandle );
	helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error
    glUseProgram ( m_programHandle );
    SetVolumeInfoUinforms ( );
	helperCheckOpenGLStatus ( __FILE__, __LINE__ );
    RenderingFace ( GL_BACK );
    glUseProgram ( 0 );
	helperCheckOpenGLStatus ( __FILE__, __LINE__ ); 

//	if ( m_density )
//		DrawDensity ( );
//	else
//		DrawVelocity ( );

	// Print FPS
	CountFPS();
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnKeyboard
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    keys, status
* @return   NULL
* @bref     Callback function, feedback for keyboard events
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnKeyboard ( SG_KEYS keys, SG_KEY_STATUS status )
{
	if ( status == SG_KEY_STATUS::SG_KEY_DOWN )
	{
		switch ( keys )
		{
		case SG_KEYS::SG_KEY_C:
			ZeroData();
			break;

		case SG_KEYS::SG_KEY_D:
			m_density = true;
			break;

		case SG_KEYS::SG_KEY_V:
			m_density = false;
			break;
		
		case SG_KEYS::SG_KEY_Q:
		case SG_KEYS::SG_KEY_ESCAPE:
			FreeResourcePtrs ( );
			OnDestroy ( );
			exit ( 0 );
			break;
		}
	}
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnMouse
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    mouse, x_pos, y_pos
* @return   NULL
* @bref     Callback function, feedback for mouse events      
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnMouse ( SG_MOUSE mouse, GLuint x_pos, GLuint y_pos )
{
#define MouseLeftDown  m_mouse->left_button_pressed
#define MouseRightDown m_mouse->right_button_pressed
#define mx	m_mouse->cur_cursor_x
#define my  m_mouse->cur_cursor_y
#define omx m_mouse->pre_cursor_x
#define omy m_mouse->pre_cursor_y

	omx = mx = x_pos;
	omx = my = y_pos;

	if ( mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN ) MouseLeftDown  = true;
	if ( mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN ) MouseRightDown = true;
	if ( mouse == SG_MOUSE::SG_MOUSE_MOVE )
	{
		mx = x_pos;
		my = y_pos;
	}

	if ( mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP ) MouseLeftDown   = false;
	if ( mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP ) MouseRightDown  = false;

#undef omx
#undef omy
#undef mx
#undef my
#undef MouseLeftDown
#undef MouseRightDown
};


/*
-----------------------------------------------------------------------------------------------------------
* @function OnDestroy
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Callback function, release the resource, exit the program when everything is done      
-----------------------------------------------------------------------------------------------------------
*/
void Visual::OnDestroy ( void )
{
	FreeResourcePtrs ( );
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	if ( cudaDeviceReset ( ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaDeviceReset was failed!" );

	SAFE_DELT_PTR ( m_mouse );
	SAFE_DELT_PTR ( m_fps );
	SAFE_DELT_PTR ( m_view );

//	if ( m_font != NULL )	m_font->Clean ( );
//	SAFE_DELT_PTR ( m_font );
};

#endif