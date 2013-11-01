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
* <Last>		Oct 25, 2013
* <File>        visual_framework.cpp
*/

#ifndef __visual_framework_cpp_
#define __visual_framework_cpp_

#include "visualFramework.h"
#include "macroDef.h"

using namespace sge;

//////////////////////////////////////////////////////////////////////////////////////////////
///

static _mouse        *m_mouse;
static _fps          *m_fps;
static _volume2D     *m_volume2D;
static _volume3D     *m_volume3D;
static _viewMatrix   *m_view;
static FreeType      *m_font;
static MainActivity  *m_hAct;
static GLfloat        m_width, m_height;
static bool           m_density;


///
//////////////////////////////////////////////////////////////////////////////////////////////
///

Visual::Visual( GLuint width, GLuint height, MainActivity *hActivity)
{
	m_mouse    = new _mouse;
	m_fps      = new _fps;
	m_volume2D = new _volume2D;
	m_volume3D = new _volume3D;
	m_view     = new _viewMatrix;
	m_font     = new FreeType;
	m_hAct     = hActivity;

	m_width    = width;
	m_height   = height;
	m_density  = false;

	m_volume2D->size = 0;
	m_volume3D->size = 0;
	
	extern void clear_data(void); extern int allocate_data(void); extern void cuda_init(void);

	// Initialize the CUDA
	cuda_init();

	if ( !allocate_data() ) exit(1);
	clear_data();
};


Visual::~Visual( void )
{
	OnDestroy();
};

///
//////////////////////////////////////////////////////////////////////////////////////////////
///

void InitFPS( void )
{
	// Zero out the frames per second variables:
	m_fps->dwFrames = 0;
	m_fps->dwCurrentTime = 0;
	m_fps->dwLastUpdateTime = 0;
	m_fps->dwElapsedTime = 0;
};


void InitFont( void )
{
	if (m_font->Init("EHSMB.TTF", 12) != SGRUNTIMEMSG::SG_RUNTIME_OK)
	{
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "Cannot init FreeType and load TTF file at line: %d of file %s", __LINE__, __FILE__);
		exit(1);
	};
}


void InitViewMatrix( void )
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


void InitMouseStatus( void )
{
	m_mouse->left_button_pressed = false;
	m_mouse->right_button_pressed = false;
};


void Setup( void )
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


void SetTexture( void )
{
	// Create 2D image texture and assign an ID
	glGenTextures(1, &m_volume2D->texture_id);

	// Create 3D image texture and assign an ID
	glGenTextures(1, &m_volume3D->texture_id);

	// Check texture ID is available
	if (m_font->IsTextureIDAvailable(m_volume2D->texture_id) != SGRUNTIMEMSG::SG_RUNTIME_OK)
	{
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "Cannot assign an available texture ID to object at line: %d of file %s", __LINE__, __FILE__);
		exit(0);
	}
};


void DrawAgent2D( void )
{
	// Bind texture
	glBindTexture(GL_TEXTURE_2D, m_volume2D->texture_id);

	// Upload 2-D textuer to OpenGL client
	glTexImage2D(GL_TEXTURE_2D,          // GLenum target
		0,		                         // GLint level,
		GL_RGB,                          // GLint internalFormat
		m_volume2D->width,               // GLsizei width
		m_volume2D->height,              // GLsizei height
		0,                               // GLint border
		GL_RGB,                          // GLenum format
		GL_UNSIGNED_BYTE,                // GLenum type
		m_volume2D->data);               // const GLvoid * data
	
	// Set texture parameters
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glPushMatrix(); // push the current matrix stack
	{
		// Rotated image first
		//glRotated(90, 0, 0, 1);

		glEnable(GL_TEXTURE_2D);  // Draw 2-D polygon agent and mapping texture onto it
		{
			glBegin(GL_QUADS);
			{
				// Front Face
				glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
				glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
				glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
				glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
			}
			glEnd();
		}
		glDisable(GL_TEXTURE_2D); // Finished
	}
	glPopMatrix();  // pop the current matrix stack
};


void CountFPS( void ) 
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


void get_from_UI(float * d, float * u, float * v)
{
#define MouseLeftDown  m_mouse->left_button_pressed
#define MouseRightDown m_mouse->right_button_pressed
#define mx	m_mouse->cur_cursor_x
#define my  m_mouse->cur_cursor_y
#define omx m_mouse->pre_cursor_x
#define omy m_mouse->pre_cursor_y

	int i, j, size = Grids_X * Grids_X;

	for (i=0 ; i<size ; i++) 
	{
		u[i] = v[i] = d[i] = 0.0f;
	}

	if (!MouseLeftDown && !MouseRightDown) return;

	i = (int)((       mx /(float)m_width)*SimArea_X+1);
	j = (int)(((m_height-my)/(float)m_height)*SimArea_X+1);

	if (i<1 || i>SimArea_X || j<1 || j>SimArea_X) return;

	if (MouseLeftDown)
	{
		u[Index(i,j)] = FORCE * (mx-omx);
		v[Index(i,j)] = FORCE * (omy-my);
	}

	if (MouseRightDown)
	{
		d[Index(i,j)] = SOURCE;
	}

	omx = mx;
	omy = my;

#undef mx
#undef my
#undef omx
#undef omy
#undef MouseLeftDown
#undef MouseRightDown
}

void free_dev_list()
{
	// Release CUDA resource if failed
	for (int i=0; i<devices; i++)
	{
		cudaFree(dev_list[i]);
	}
	dev_list.empty();
}

void cuda_init()
{	
	// Push dev into vector
	for (int i=0; i<devices; i++)
	{
		static float *ptr;
		dev_list.push_back(ptr);
	}

	size_t size = Grids_X * Grids_X;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "cudaSetDevice was failed, do you have a CUDA-capable GPU installed? at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString(cudaStatus));
        free_dev_list();
    }

    // Allocate GPU buffers for three vectors (two input, one output).
	for (int i=0; i<devices; i++)
	{
		cudaStatus = cudaMalloc((void**)&dev_list[i], size * sizeof(float));
		if (cudaStatus != cudaSuccess) {
			Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "cudaMalloc was failed, at line: %d of file %s", __LINE__, __FILE__);
			Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString(cudaStatus));
			free_dev_list();
		}
	}
};

void free_data(void)
{
	if ( u ) SAFE_FREE_PTR(u);
	if ( v ) SAFE_FREE_PTR(v);
	if ( u_prev ) SAFE_FREE_PTR(u_prev);
	if ( v_prev ) SAFE_FREE_PTR(v_prev);
	if ( dens ) SAFE_FREE_PTR(dens);
	if ( dens_prev ) SAFE_FREE_PTR(dens_prev);

	// Release CUDA resources
	for (int i=0; i<devices; i++)
		cudaFree(dev_list[i]);

}

void clear_data(void)
{
	int size=(Grids_X)*(Grids_X);

	for ( int i=0; i<size ; i++ )
	{
		u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
	}
}

int allocate_data(void)
{
	int size = (Grids_X)*(Grids_X);

	u			= (float *)malloc(size*sizeof(float));
	v			= (float *)malloc(size*sizeof(float));
	u_prev		= (float *)malloc(size*sizeof(float));
	v_prev		= (float *)malloc(size*sizeof(float));
	dens		= (float *)malloc(size*sizeof(float));	
	dens_prev	= (float *)malloc(size*sizeof(float));

	if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) 
	{
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "allocate data was failed, at line: %d of file %s", __LINE__, __FILE__);
		return ( 0 );
	}

	return 1;
}


///
//////////////////////////////////////////////////////////////////////////////////////////////
///

void Visual::OnCreate( void )
{
	// Initialize
	InitViewMatrix();
	InitFont();
	InitFPS();
	InitMouseStatus();

	// Initialize glew
	glewInit();

	// Call for OpenGL envrionment setup
//	Setup();

	// Set texture
//	SetTexture();
	
};


void Visual::OnResize( GLuint width, GLuint height )
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

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);
	glClearColor( 0.0f, 0.0f, 0.0f, 1.0f);
};


void Visual::OnIdle( void )
{
	extern void get_from_UI(float *, float *, float *);

	get_from_UI(dens_prev, u_prev, v_prev);
	vel_step(u, v, u_prev, v_prev);
	dens_step(dens, dens_prev, u, v);

};


void Visual::OnDisplay( void )
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
	extern void draw_density(), draw_velocity();

	glClear(GL_COLOR_BUFFER_BIT);

	if (m_density)
		draw_density();
	else
		draw_velocity();

	// Print FPS
	CountFPS();
};


void Visual::OnKeyboard( SG_KEYS keys, SG_KEY_STATUS status )
{
	extern void clear_data(), free_data();

	if (status == SG_KEY_STATUS::SG_KEY_DOWN)
	{
		switch (keys)
		{
		case SG_KEYS::SG_KEY_C:
			clear_data();
			break;

		case SG_KEYS::SG_KEY_D:
			m_density = true;
			break;

		case SG_KEYS::SG_KEY_V:
			m_density = false;
			break;
		
		case SG_KEYS::SG_KEY_Q:
		case SG_KEYS::SG_KEY_ESCAPE:
			free_data();
			OnDestroy();
			exit ( 0 );
			break;
		}
	}
};


void Visual::OnMouse( SG_MOUSE mouse, GLuint x_pos, GLuint y_pos )
{
#define MouseLeftDown  m_mouse->left_button_pressed
#define MouseRightDown m_mouse->right_button_pressed
#define mx	m_mouse->cur_cursor_x
#define my  m_mouse->cur_cursor_y
#define omx m_mouse->pre_cursor_x
#define omy m_mouse->pre_cursor_y

	omx = mx = x_pos;
	omx = my = y_pos;

	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN) MouseLeftDown  = true;
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN) MouseRightDown = true;
	if (mouse == SG_MOUSE::SG_MOUSE_MOVE)
	{
		mx = x_pos;
		my = y_pos;
	}

	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP) MouseLeftDown   = false;
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP) MouseRightDown  = false;

#undef omx
#undef omy
#undef mx
#undef my
#undef MouseLeftDown
#undef MouseRightDown
};


void Visual::OnDestroy( void )
{
	extern void free_data();

	free_data();
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
		Logfile.SaveStringToFile("errormsg.log", SG_FILE_OPEN_APPEND, "cudaDeviceReset was failed, at line: %d of file %s", __LINE__, __FILE__);
		Logfile.SaveStringToFile("errormsg.log", sge::SG_FILE_OPEN_APPEND, ">>>> Error Message: %s", cudaGetErrorString(cudaStatus));
    }

	SAFE_DELT_PTR( m_mouse );
	SAFE_DELT_PTR( m_fps );
	SAFE_DELT_PTR( m_view );

	if ( m_volume2D->size > 0 ) SAFE_FREE_PTR( m_volume2D->data );
	if ( m_volume3D->size > 0 ) SAFE_FREE_PTR( m_volume3D->data );

	if ( m_font != NULL )	m_font->Clean();
	SAFE_DELT_PTR( m_font );
};

///
//////////////////////////////////////////////////////////////////////////////////////////////
///

void Visual::UploadVolumeData( _volume2D const *data_in )
{
	m_volume2D->width  = data_in->width;
	m_volume2D->height = data_in->height;
	m_volume2D->data   = data_in->data;

	m_volume2D->size   = data_in->size;
};


void Visual::UploadVolumeData( _volume3D const *data_in )
{
	m_volume3D->width  = data_in->width;
	m_volume3D->height = data_in->height;
	m_volume3D->depth  = data_in->depth;
	m_volume3D->data   = data_in->data;

	m_volume3D->size   = data_in->size;
};

int Visual::Texel2D( int i, int j )
{
	return  BYTES_PER_TEXEL * ( i * m_volume2D->height + j );
};


int Layer( int layer )
{
	return layer * m_volume3D->width * m_volume3D->height;
};


int Visual::Texel3D( int i, int j, int k )
{
	return BYTES_PER_TEXEL * ( Layer( i ) + m_volume3D->height * j + k);
};

///
//////////////////////////////////////////////////////////////////////////////////////////////

#endif