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
* <Last>		Nov 6, 2013
* <File>        visual_framework.cpp
*/

#ifndef __visual_framework_cpp_
#define __visual_framework_cpp_

#define _CRT_SECURE_NO_WARNINGS


#include "cfdHeader.h"

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

#include "GLSLHelper.h"

using namespace std;
using namespace sge;


// shader and texture handle
static GLuint         m_tffTexObj;    /* 1-D ray casting texture */
static GLuint         m_bfTexObj;     /* 2-D backface texture */
static GLuint         m_volTexObj;    /* 3-D volume data texture */
static GLuint         m_rcVertHandle; /* vertex shader of rc */
static GLuint         m_rcFragHandle; /* fragment shader of rc */
static GLuint         m_bfVertHandle; /* vertex shader of bf */
static GLuint         m_bfFragHandle; /* fragment shader of bf */


static _mouse        *m_mouse;
static _fps          *m_fps;
static _viewMatrix   *m_view;
static GLint          m_width, m_height;
static bool           m_density;
static GLuint         m_cluster;
static GLuint         m_programHandle;
static GLint          m_angle = 0;
static GLuint         m_frameBuffer;



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

	extern SGRUNTIMEMSG AllocateResourcePtrs ();
	extern void FreeResourcePtrs ();
	extern void ZeroData ();

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



GLuint initTFF1DTex ( const char* filename )
{
    // Read in the user defined data of transfer function
    ifstream inFile ( filename, ifstream::in );
	if ( !inFile )
    {
		cerr << "Error openning file: " << filename << endl;
		exit(EXIT_FAILURE);
    }
    
    const int MAX_CNT = 10000;
    GLubyte *tff = (GLubyte *) calloc(MAX_CNT, sizeof(GLubyte));
    inFile.read ( reinterpret_cast<char *>(tff), MAX_CNT );

	// File is end?
    if (inFile.eof())
    {
		size_t bytecnt = inFile.gcount();
		*(tff + bytecnt) = '\0';
		cout << "bytecnt " << bytecnt << endl;
	}
    else if(inFile.fail())
    {
		cout << filename << "read failed " << endl;
    }
    else
    {
		cout << filename << "is too large" << endl;
    }
	
	GLuint tff1DTex;
    glGenTextures(1, &tff1DTex);
    glBindTexture(GL_TEXTURE_1D, tff1DTex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, tff);
    
	free(tff);    
    
	return tff1DTex;
};


GLuint initFace2DTex ( GLuint bfTexWidth, GLuint bfTexHeight )
{
    GLuint backFace2DTex;
    glGenTextures(1, &backFace2DTex);
    glBindTexture(GL_TEXTURE_2D, backFace2DTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, bfTexWidth, bfTexHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	return backFace2DTex;
};


GLuint initVol3DTex ( const char* filename, GLuint w, GLuint h, GLuint d )
{
    FILE *fp;
    size_t size = w * h * d;
    GLubyte *data = new GLubyte[size];			  // 8bit
 
	if ( !(fp = fopen(filename, "rb")) )
    {
        cout << "Error: opening .raw file failed" << endl;
        exit ( EXIT_FAILURE );
    }
    else
    {
        cout << "OK: open .raw file successed" << endl;
    }
    if ( fread(data, sizeof(char), size, fp)!= size) 
    {
        cout << "Error: read .raw file failed" << endl;
        exit ( 1 );
    }
    else
    {
        cout << "OK: read .raw file successed" << endl;
    }
    fclose ( fp );

	// Generate 3D textuer
    glGenTextures(1, &m_volTexObj);
    glBindTexture(GL_TEXTURE_3D, m_volTexObj);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
    
	// pixel transfer happens here from client to OpenGL server
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, w, h, d, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE,data);

    delete []data;
    cout << "volume texture created" << endl;
    return m_volTexObj;
};


void initFrameBuffer ( GLuint texObj, GLuint texWidth, GLuint texHeight )
{
    // Create a depth buffer for framebuffer
    GLuint depthBuffer;
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, texWidth, texHeight);

    // Attach the texture and the depth buffer to the framebuffer
    glGenFramebuffers(1, &m_frameBuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, m_frameBuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texObj, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	
	// Check Framebuffer status
	if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE )
    {
		cout << "framebuffer is not complete" << endl;
		exit(EXIT_FAILURE);
    }
    glEnable(GL_DEPTH_TEST);    
};


void initShader ( void )
{
	// To create backface shader for first pass
	helperInitPairShadersObj ( 
		&m_bfVertHandle, "shader/backface.vert", 
		&m_bfFragHandle, "shader/backface.frag" );

	// To create raycasting shader for second pass
	helperInitPairShadersObj ( 
		&m_rcVertHandle, "shader/raycasting.vert", 
		&m_rcFragHandle, "shader/raycasting.frag" );

	// Link shader program
	helperCreateShaderProgram ( &m_programHandle );
};


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
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_3D, m_volTexObj);
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
    helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error
    initFrameBuffer ( m_bfTexObj, m_width, m_height );
    helperCheckOpenGLStatus ( __FILE__, __LINE__ ); // Check OpenGL runtime error
};


/*
-----------------------------------------------------------------------------------------------------------
* @function CoutFPS
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     When flush the image onto screen, call this function to count FPS and update its status      
-----------------------------------------------------------------------------------------------------------
*/
void CountFPS ( void ) 
{
	// Calculate the number of frames per one second:
	m_fps->dwFrames++;
	m_fps->dwCurrentTime = GetTickCount(); // Even better to use timeGetTime()
	m_fps->dwElapsedTime = m_fps->dwCurrentTime - m_fps->dwLastUpdateTime;
	
	// Already 1s
	if ( m_fps->dwElapsedTime >= 1000 )
	{
		m_fps->FPS = m_fps->dwFrames * 1000.0 / m_fps->dwElapsedTime;
		m_fps->dwFrames = 0;
		m_fps->dwLastUpdateTime = m_fps->dwCurrentTime;
	}

//	glPushMatrix();
//	{
//		glLoadIdentity();									// Reset The Current Modelview Matrix
//		glTranslatef(0.0f,0.0f,-1.0f);						// Move One Unit Into The Screen
//		
//		// White Text
//		glColor3f(0.0f, 1.0f, 0.0f);
//		m_font->EnableFreeType();
//		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", m_fps->FPS);
//		m_font->DisableFreeType();
//	}
//	glPopMatrix();
	
	system ("cls");
	printf ("Current's FPS:   %d", m_fps->FPS);
}


/*
-----------------------------------------------------------------------------------------------------------
* @function FreeResourcePtrs
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Release resource      
-----------------------------------------------------------------------------------------------------------
*/
void FreeResourcePtrs ( void )
{
	// Release ptr of host
	for ( int i = 0; i < HostListNum; i++ )
	{
		if ( host_list [ i ] ) SAFE_FREE_PTR ( host_list [ i ] );
	}
	host_list.empty ( );

	// Release ptr of CUDA
	for ( int i = 0; i < DevListNum; i++ )
	{
		cudaFree ( dev_list [ i ] );
	}
	dev_list.empty ( );

	// Release other resource
	for ( int i = 0; i < BufferHostListNum; i++ )
	{
		if ( buffer_host_list [ i ] ) SAFE_FREE_PTR ( buffer_host_list [ i ] );
	}
	buffer_host_list.empty ( );

	// And ...
	for ( int i = 0; i < BufferDeviceListNum; i++ )
	{
		cudaFree ( buffer_dev_list [ i ] );
	}
	buffer_dev_list.empty ( );
}


/*
-----------------------------------------------------------------------------------------------------------
* @function ZeroData
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Zero the buffers
-----------------------------------------------------------------------------------------------------------
*/
void ZeroData ( void )
{
	for ( int i = 0; i < SIM_SIZE; i++ )
	{
		host_u [ i ] = 0.f;
		host_v [ i ] = 0.f;
		host_w [ i ] = 0.f;
		host_u0 [ i ] = 0.f;
		host_v0 [ i ] = 0.f;
		host_w0 [ i ] = 0.f;
		host_den [ i ] = 0.f;
		host_den0 [ i ] = 0.f;
	}
}


/*
-----------------------------------------------------------------------------------------------------------
* @function AllocateResourcePtrs
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Allocate resource      
-----------------------------------------------------------------------------------------------------------
*/
SGRUNTIMEMSG AllocateResourcePtrs ( void )
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	if ( cudaSetDevice ( 0 ) != cudaSuccess )
		cudaCheckRuntimeErrors ( "cudaSetDevices" );

	// Allocate memory on Host
	for ( int i = 0; i < HostListNum; i++ )
	{
		static float *ptr;
		ptr = ( float * ) malloc ( SIM_SIZE*sizeof ( float ) );
		host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! host_list [ i ] )
		{
			Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND,
				"allocate data was failed, at line: %d of file %s", __LINE__, __FILE__ );
			return SG_RUNTIME_FALSE;
		}
	}

	// Allocate memory on GPU devices
	for ( int i = 0; i < DevListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( ( void ** ) &ptr, SIM_SIZE * sizeof ( float ) ) != cudaSuccess )
		{
			cudaCheckRuntimeErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		dev_list.push_back(ptr);
	}

	// Allocate rendering buffers both on host and GPU devices
	// Host first
	for ( int i = 0; i < BufferHostListNum; i++ )
	{
		static float *ptr;
		ptr = ( float * ) malloc ( DIS_SIZE * sizeof ( float ) );
		buffer_host_list.push_back ( ptr );

		// Alarm if null pointer
		if ( ! buffer_host_list [ i ] )
		{
			Logfile.SaveStringToFile ( "errormsg.log", SG_FILE_OPEN_APPEND,
				"allocate data was failed, at line: %d of file %s", __LINE__, __FILE__ );
			return SG_RUNTIME_FALSE;
		}
	}

	// Then GPU devices
	for ( int i = 0; i < BufferDeviceListNum; i++ )
	{
		// Alarm if cudaMalloc failed
		static float *ptr;
		if ( cudaMalloc( ( void ** ) &ptr, DIS_SIZE * sizeof ( float ) ) != cudaSuccess )
		{
			cudaCheckRuntimeErrors ( "cudaMalloc failed!" );
			return SG_RUNTIME_FALSE;
		}
		buffer_dev_list.push_back(ptr);
	}

	// Finally
	return SG_RUNTIME_OK;
}


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
//	VelocitySolver ( host_u, host_v, host_w, host_u0, host_v0, host_w0 );
//	DensitySolver ( host_den, host_den0, host_u, host_v, host_w );
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
//
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