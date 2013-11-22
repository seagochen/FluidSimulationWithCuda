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

#include "cfdHeader.h"

using namespace sge;

static _mouse        *m_mouse;
static _fps          *m_fps;
static _viewMatrix   *m_view;
static FreeType      *m_font;
static MainActivity  *m_hAct;
static GLfloat        m_width, m_height;
static bool           m_density, m_macCormack;


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
	if (m_font->Init("EHSMB.TTF", 12) != SGRUNTIMEMSG::SG_RUNTIME_OK)
	{
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, 
			"Cannot init FreeType and load TTF file at line: %d of file %s",
			__LINE__, __FILE__);
		exit(1);
	};
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
	m_mouse->left_button_pressed = false;
	m_mouse->right_button_pressed = false;
};


/*
-----------------------------------------------------------------------------------------------------------
* @function Setup
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     Initialize OpenGL, Rendering, the model of view matrix and etc.
-----------------------------------------------------------------------------------------------------------
*/
void Setup ( void )
{
	// Enable depth testing
	glEnable(GL_DEPTH_TEST);

	// Enable clearing of the depth buffer
	glClearDepth(1.f);

	// Type of depth test to do
	glDepthFunc(GL_LEQUAL);	

	// Specify implementation-specific hints
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
	
	// Enable smooth color shading
	glShadeModel(GL_SMOOTH);

	// Changing matrix
	glMatrixMode(GL_PROJECTION);

	// Reset the projection matrix
	glLoadIdentity();

	// Calculate the aspect ratio of the window
	gluPerspective(m_view->view_angle, m_width / m_height, m_view->z_near, m_view->z_far);

	// Changing matrix 
	glMatrixMode(GL_MODELVIEW);

	// Set clearing color
	glClearColor(0.f, 0.f, 0.0f, 1.f);
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

	glPushMatrix();
	{
		glLoadIdentity();									// Reset The Current Modelview Matrix
		glTranslatef(0.0f,0.0f,-1.0f);						// Move One Unit Into The Screen
		
		// White Text
		glColor3f(0.0f, 1.0f, 0.0f);
		m_font->EnableFreeType();
		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", m_fps->FPS);
		m_font->DisableFreeType();
	}
	glPopMatrix();
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

	// Initialize glew
	glewInit ( );

	// Call for OpenGL envrionment setup
//	Setup();

	// Set texture
//	SetTexture();
	
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
	// Prevent a divide by zero if the window is too small
//	if (height == 0) height = 1;

	m_width  = width;
	m_height = height;

//	glViewport(0, 0, width, height);

	// Reset the current viewport and perspective transformation
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	gluPerspective(m_view->view_angle, m_width / m_height, m_view->z_near, m_view->z_far);
//	glMatrixMode(GL_MODELVIEW);

	if (height == 0) height = 1;

	glViewport(0, 0, width, height);

	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ( );
	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
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
	if ( m_macCormack == false )
	{
		VelocitySolver ( host_u, host_v, host_w, host_u0, host_v0, host_w0 );
		DensitySolver ( host_den, host_den0, host_u, host_v, host_w );
	}
	else
	{
		MacCormackSchemeSolver ( host_u, host_v, host_w, host_u0, host_v0, host_w0, host_den, host_den0 );
	}
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
	// Reset matrix
//	glLoadIdentity();

	// Clear Screen and Depth Buffer
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set camera
//	gluLookAt(
//		m_view->eye_x,  m_view->eye_y,  m_view->eye_z,  // eye
//		m_view->look_x, m_view->look_y, m_view->look_z, // center
//		m_view->up_x,   m_view->up_y,   m_view->up_z);  // Up

	// Draw fluid sim result on 2-D map
//	DrawAgent2D();

	glClear ( GL_COLOR_BUFFER_BIT );

	if ( m_density )
		DrawDensity ( );
	else
		DrawVelocity ( );

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

		case SG_KEYS::SG_KEY_M:
			ZeroData ();
			m_macCormack = true;
			break;

		case SG_KEYS::SG_KEY_N:
			ZeroData ();
			m_macCormack = false;
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

	if ( m_font != NULL )	m_font->Clean ( );
	SAFE_DELT_PTR ( m_font );
};


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
Visual::Visual ( GLuint width, GLuint height, MainActivity *hActivity )
{
	// Materialized , and the assignment
	m_mouse    = new _mouse;
	m_fps      = new _fps;
	m_view     = new _viewMatrix;
	m_font     = new FreeType;
	m_hAct     = hActivity;

	m_width    = width;
	m_height   = height;
	m_density  = false;
	m_macCormack = false;

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
#endif