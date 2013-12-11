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
* <Last>		Nov 10, 2013
* <File>        initParam.cpp
*/

#ifndef __init_param_cpp_
#define __init_param_cpp_

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

using namespace sge;

/*
-----------------------------------------------------------------------------------------------------------
* @function InitFPS
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Initialization parameters associated with FPS   
-----------------------------------------------------------------------------------------------------------
*/
void InitFPS ( void )
{
	// Zero out the frames per second variables:
	m_fps->dwFrames = 0;
	m_fps->dwCurrentTime = 0;
	m_fps->dwLastUpdateTime = 0;
	m_fps->dwElapsedTime = 0;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function InitFont
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Initialization parameters associated with FreeType Font     
-----------------------------------------------------------------------------------------------------------
*/
void InitFont ( void )
{
	// Initialize the font source, otherwise arise the exception
//	if ( m_font->Init ( "EHSMB.TTF", 12 ) != SGRUNTIMEMSG::SG_RUNTIME_OK )
//	{
//		Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND, 
//			"Cannot init FreeType and load TTF file at line: %d of file %s",
//			__LINE__, __FILE__ );
//		exit ( 1 );
//	};
}


/*
-----------------------------------------------------------------------------------------------------------
* @function InitViewMatrix
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Initialization parameters associated with observation Matrix, for 3-D display
-----------------------------------------------------------------------------------------------------------
*/
void InitViewMatrix ( void )
{
	// view matrix
	m_view->view_angle    = 45.f;
	// eye
	m_view->eye_x         = 0.f;
	m_view->eye_y         = 0.f;
	m_view->eye_z         = 3.f;
	// look at
	m_view->look_x        = 0.f;
	m_view->look_y        = 0.f;
	m_view->look_z        = 0.f;
	// up
	m_view->up_x          = 0.f;
	m_view->up_y          = 1.f;
	m_view->up_z          = 0.f;
	// near & far
	m_view->z_far         = 100.f;
	m_view->z_near        = 0.1f;
	// forward
	m_view->z_forward     = -5.f;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function InitMouseStatus
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Initialization parameters associated with mouse      
-----------------------------------------------------------------------------------------------------------
*/
void InitMouseStatus ( void )
{
	m_mouse->left_button_pressed  = false;
	m_mouse->right_button_pressed = false;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function Setup
* @author   Orlando Chen
* @date     Dce 4, 2013
* @input    NULL
* @return   NULL
* @bref     Initialize OpenGL, Rendering, the model of view matrix and etc.
-----------------------------------------------------------------------------------------------------------
*/
void Setup ( void )
{
	// Initialize glew first
	GLenum err = glewInit ( );
	if ( GLEW_OK != err )
	{
		Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND, 
			"Cannot init glewInit, error: %s at line: %d of file %s",
			glewGetErrorString ( err ), __LINE__, __FILE__ );
		exit ( 1 );
	}

	helperInitVerticesBufferObj ( &m_cluster );
    initShader ( );
    m_tffTexObj = initTFF1DTex  ( "tff.dat" );
	m_bfTexObj  = initFace2DTex ( m_width, m_height );
    m_volTexObj = initVol3DTex  ( "head256.raw", 256, 256, 225 );
};

#endif