#include "Visualization.h"
#include <Eigen\Dense>
#include <iostream>
#include <Winuser.h>

using namespace sge;


#ifndef _SGE_DEFINED_VALUE_
#define _SGE_DEFINED_VALUE_

static _mouse      *m_mouse;
static _fps        *m_fps;
static _volumeData *m_volume;
static _viewMatrix *m_view;
static FreeType    *m_font;
static MainActivity *m_hAct;

#endif

#define pstatus(str) {system("cls"); printf("Status: %s \n", str);}
#define PI 3.14159
#define sqr(num) pow(num, 2)

Visualization::Visualization(MainActivity *hActivity)
{
	m_mouse  = new _mouse;
	m_fps    = new _fps;
	m_volume = new _volumeData;
	m_view   = new _viewMatrix;
	m_font   = new FreeType;

	m_hAct   = hActivity;
};


Visualization::~Visualization()
{
	SAFE_DELT_PTR(m_mouse);
	SAFE_DELT_PTR(m_fps);
	SAFE_FREE_PTR(m_volume->data);
	SAFE_DELT_PTR(m_volume);
	SAFE_DELT_PTR(m_view);
	m_font->Clean();
	SAFE_DELT_PTR(m_font)
};


void Visualization::GetActivityHandler(MainActivity *hActivity)
{
	m_hAct = hActivity;
};


void Visualization::InitFPS()
{
	// Zero out the frames per second variables:
	m_fps->dwFrames = 0;
	m_fps->dwCurrentTime = 0;
	m_fps->dwLastUpdateTime = 0;
	m_fps->dwElapsedTime = 0;
};


void Visualization::InitFont()
{
	m_font->Init("EHSMB.TTF", 12);
}


void Visualization::InitViewMatrix()
{
	// view matrix
	m_view->field_of_view_angle    = 45.f;
	// view radius
	m_view->radius_of_view_matrix  = 2.f;
	// eye
	m_view->eye_at_x               = 0.f;
	m_view->eye_at_y               = 0.f;
	m_view->eye_at_z               = 5.f;
	// look at
	m_view->look_at_x              = 0.f;
	m_view->look_at_y              = 0.f;
	m_view->look_at_z              = 0.f;
	// direct up
	m_view->dx_up_x                = 0.f;
	m_view->dx_up_y                = 1.f;
	m_view->dx_up_z                = 0.f;
	// near & far
	m_view->z_far                  = 100.f;
	m_view->z_near                 = 0.1f;
	// forward
	m_view->z_forward              = -5.f;
	// rotation
	m_view->rotate_of_x            = 0.f;
	m_view->rotate_of_y            = 0.f;
	m_view->rotate_of_z            = 0.f;
};


void Visualization::Bind3DTexutre()
{
	// 创建3D纹理
	glGenTextures(1, &m_volume->textureID);
	glBindTexture(GL_TEXTURE_3D, m_volume->textureID);

	// 设定贴图参数
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	// 载入3D纹理，并设置相关参数
	glTexImage3D(GL_TEXTURE_3D,          // GLenum target
		0,		                         // GLint level,
		GL_RGB,                          // GLint internalFormat
		m_volume->width,                 // GLsizei width
		m_volume->height,                // GLsizei height
		m_volume->depth,                 // GLsizei depth
		0,                               // GLint border
		GL_RGB,                          // GLenum format
		GL_UNSIGNED_BYTE,                // GLenum type
		m_volume->data);                 // const GLvoid * data
};


void EnableFunc(void)
{
	// Enable blend effect
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

	// Enable depth testing
	glEnable(GL_DEPTH_TEST);

	// Enable culll face
	glEnable(GL_CULL_FACE);
}


void Visualization::Init(GLuint width, GLuint height)
{
	// Initialize
	glewInit();
	InitFont();
	InitFPS();
	InitViewMatrix();

	// Enable
	EnableFunc();

	// Load the textures
	Bind3DTexutre();

	// Set clearing color
	glClearColor(0.f, 0.f, 0.0f, 1.f);

	// Enable clearing of the depth buffer
	glClearDepth(1.f);

	// Type of depth test to do
	//glDepthFunc(GL_LESS);
	glDepthFunc(GL_LEQUAL);	

	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// Enable smooth color shading
	glShadeModel(GL_SMOOTH);

	// Changing matrix
	glMatrixMode(GL_PROJECTION);

	// Reset the projection matrix
	glLoadIdentity();

	// Calculate the aspect ratio of the window
	gluPerspective(m_view->field_of_view_angle,
		(GLfloat)width / (GLfloat)height, m_view->z_near, m_view->z_far);

	// Changing matrix 
	glMatrixMode(GL_MODELVIEW);

	m_mouse->left_button_pressed = false;
	m_mouse->right_button_pressed = false;
}


void Visualization::ResizeScreen(GLuint width, GLuint height)
{
	// Prevent a divide by zero if the window is too small
	if (height == 0) height = 1;

	glViewport(0, 0, width, height);

	// Reset the current viewport and perspective transformation
	glMatrixMode(GL_PROJECTION);
	{
		glLoadIdentity();
		
		gluPerspective(m_view->field_of_view_angle,
			(GLfloat)width / (GLfloat)height, m_view->z_near, m_view->z_far);
	}
	glMatrixMode(GL_MODELVIEW);
}


void Visualization::Display() 
{
	glLoadIdentity();
	// Clear Screen and Depth Buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// 通过移动Camera实现对模型的观察
	gluLookAt(
		m_view->eye_at_x,  m_view->eye_at_y,  m_view->eye_at_z,  // eye
		m_view->look_at_x, m_view->look_at_y, m_view->look_at_z, // center
		m_view->dx_up_x,   m_view->dx_up_y,   m_view->dx_up_z);  // Up
	
	// 放置代理幾何
	glPushMatrix();
	{
		// 绘制代理几何
		glEnable(GL_TEXTURE_3D);
		{
			CreateAgentBox();
		}
		glDisable(GL_TEXTURE_3D);
	}
	glPopMatrix();

	CountFPS();
}


void Visualization::CreateAgentBox()
{
	// 将数据实时显示在屏幕上，这里采用glTexSubImage3D而不是glTexImage3D，是为了降低频繁更新而带来的性能损耗
	// every 10 frames per second
	if (m_fps->dwElapsedTime > 500)
	{
		glTexSubImage3D(
			GL_TEXTURE_3D,                 // GLenum target,
			0,                             // GLint level,
			0,                             // GLint xoffset,
			0,                             // GLint yoffset,
			0,                             // GLint zoffset,
			m_volume->width,               // GLsizei width,
			m_volume->height,              // GLsizei height,
			m_volume->depth,               // GLsizei depth,
			GL_RGB,                        // GLenum format,
			GL_UNSIGNED_BYTE,              // GLenum type,
			m_volume->data);               // const GLvoid * data
	}

	// Draw 6 quadrilaterals
	glBegin(GL_QUADS);
	{
		// Front Face
		glTexCoord3f(0.0f, 0.0f,1.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(1.0f, 0.0f,1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Top Right Of The Texture and Quad
		glTexCoord3f(0.0f, 1.0f,1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Top Left Of The Texture and Quad
		
		// Back Face
		glTexCoord3f(0.0f, 0.0f,0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(0.0f, 1.0f,0.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,0.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord3f(1.0f, 0.0f,0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Bottom Left Of The Texture and Quad
 
		// Top Face
		glTexCoord3f(0.0f, 1.0f,0.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord3f(0.0f, 1.0f,1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,0.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
 
		// Bottom Face
		glTexCoord3f(0.0f, 0.0f,0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord3f(1.0f, 0.0f,0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord3f(1.0f, 0.0f,1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(0.0f, 0.0f,1.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
 
		// Right face
		glTexCoord3f(1.0f, 0.0f,0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,0.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord3f(1.0f, 1.0f,1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Top Left Of The Texture and Quad
		glTexCoord3f(1.0f, 0.0f,1.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
 
		// Left Face
		glTexCoord3f(0.0f, 0.0f,0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord3f(0.0f, 0.0f,1.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord3f(0.0f, 1.0f,1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Top Right Of The Texture and Quad
		glTexCoord3f(0.0f, 1.0f,0.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
	}
	glEnd();
};


void Visualization::CountFPS( ) {
	
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
		glColor3f(1.0f, 1.0f, 1.0f);
		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", m_fps->FPS);			
	}
	glPopMatrix();

}


int Visualization::LoadVolumeData(_volumeData const *data_in)
{
	if (data_in == NULL) 
	{
		pterror("Error you've passed an empty pointer");
		return SG_FAIL;
	}

	memcpy(m_volume, data_in, sizeof(_volumeData));

	if ( m_volume->data != NULL ) return SG_OK;
	
	return SG_FAIL;
};


int Visualization::SaveVolumeData(_volumeData *data_out)
{
	if (m_volume->data == NULL) 
	{
		pterror("Error you've tried access an empty pointer");
		return SG_FAIL;
	}

	memcpy(data_out, m_volume, sizeof(_volumeData));

	if ( data_out->data != NULL ) return SG_OK;
	
	return SG_FAIL;
}


int Visualization::SetWindowParam(_viewMatrix const *view_matrix_in)
{
	if (view_matrix_in == NULL)
	{
		pterror("Error you've passed an empty pointer");
		return SG_FAIL;
	}

	memcpy(m_view, view_matrix_in, sizeof(_viewMatrix));

	if (m_view == NULL) return SG_FAIL;

	return SG_OK;
};


int Visualization::GetWindowParam(_viewMatrix *view_matrix_out)
{
	if (m_view == NULL) 
	{
		pterror("Error you've tried access an empty pointer");
		return SG_FAIL;
	}

	memcpy(view_matrix_out, m_view, sizeof(_viewMatrix));

	if (view_matrix_out == NULL) return SG_FAIL;

	return SG_OK;
};


void AwayObjects()
{
	Eigen::Vector3f v3Eyef(m_view->eye_at_x, m_view->eye_at_y, m_view->eye_at_z);
	Eigen::Vector3f v3Orgf(0, 0, 0);
	Eigen::Vector3f v3Dirf = v3Eyef - v3Orgf;
	v3Dirf.normalize();
	v3Dirf *= 0.1f;
	v3Eyef += v3Dirf;

	// Applied new position
	m_view->eye_at_x = v3Eyef[0];
	m_view->eye_at_y = v3Eyef[1];
	m_view->eye_at_z = v3Eyef[2];

#ifdef _DEBUG
	std::cout<< v3Eyef <<std::endl;
#endif
};


void ApproachObjects()
{
	Eigen::Vector3f v3Eyef(m_view->eye_at_x, m_view->eye_at_y, m_view->eye_at_z);
	Eigen::Vector3f v3Orgf(0, 0, 0);
	Eigen::Vector3f v3Dirf = v3Eyef - v3Orgf;
	v3Dirf.normalize();
	v3Dirf *= 0.1f;
	v3Eyef -= v3Dirf;

	// Applied new position
	m_view->eye_at_x = v3Eyef[0];
	m_view->eye_at_y = v3Eyef[1];
	m_view->eye_at_z = v3Eyef[2];

#ifdef _DEBUG
	std::cout<< v3Eyef <<std::endl;
#endif
};


void RotateObjects(unsigned x, unsigned y)
{
	if(m_mouse->left_button_pressed)
	{
		int xDist = x - m_mouse->pre_cursor_x;
		int yDist = y - m_mouse->pre_cursor_y;

#ifdef _DEBUG
		std::cout<<x<<" - "<<m_mouse->pre_cursor_x<<" = "<<xDist<<std::endl;
		std::cout<<y<<" - "<<m_mouse->pre_cursor_y<<" = "<<yDist<<std::endl;
#endif

		m_mouse->pre_cursor_x = x;
		m_mouse->pre_cursor_y = y;
		
		if (xDist > 0)
			m_view->rotate_of_x += 0.1f;
		else if (xDist < 0)
			m_view->rotate_of_x -= 0.1f;

		if (yDist > 0)
			m_view->rotate_of_y += 0.1f;
		else if (xDist < 0)
			m_view->rotate_of_y -= 0.1f;

#ifdef _DEBUG
		std::cout<<"X-axis"<<m_view->rotate_of_x * 0.1<<"'"<<std::endl;
		std::cout<<"Y-axis"<<m_view->rotate_of_y * 0.1<<"'"<<std::endl;
#endif

		Eigen::Vector3f eye(m_view->eye_at_x, m_view->eye_at_y, m_view->eye_at_z);
		float radius = sqr(eye[0]) + sqr(eye[1]) + sqr(eye[2]);
		radius = sqrt(radius);
		
		m_view->eye_at_x = cos( m_view->rotate_of_x ) * radius;
		m_view->eye_at_z = sin( m_view->rotate_of_x ) * radius;

		eye[0] = m_view->eye_at_x;
		eye[1] = m_view->eye_at_y;
		eye[2] = m_view->eye_at_z;

#ifdef _DEBUG
		std::cout<<eye<<std::endl;
#endif
	}

#ifdef __USING
	// Get vectors of direction up right
	system("cls");
	Eigen::Vector3f lookat(cos( m_view->rotate_of_x ), 0.f, sin( m_view->rotate_of_x ));
	Eigen::Vector3f right(cos( m_view->rotate_of_x + PI/2), 0.f, sin( m_view->rotate_of_x + PI/2));
	Eigen::Vector3f up = right.cross(lookat);
#endif 
};


void MouseMotion(SG_MOUSE mouse, unsigned x, unsigned y)
{
	// Away from the object
	if (mouse == SG_MOUSE::SG_MOUSE_WHEEL_BACKWARD)
	{
		AwayObjects();
	}

	// Approaching object
	if (mouse == SG_MOUSE::SG_MOUSE_WHEEL_FORWARD)
	{
		ApproachObjects();
	}
	
	// Mouse
	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN || mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN)
	 {
		 m_mouse->pre_cursor_x = x;
		 m_mouse->pre_cursor_y = y;
		 
		 if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN)
			 m_mouse->left_button_pressed = true;
		 else if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN)
			 m_mouse->right_button_pressed = true;
	 }
	 else if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP || mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP)
	 {
		 if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP) 
			 m_mouse->left_button_pressed = false;
		 else if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP)
			 m_mouse->right_button_pressed = false;
	 }

	// Mouse motion
	if (mouse == SG_MOUSE::SG_MOUSE_MOVE)
	{
		RotateObjects(x, y);
	}
};


void Visualization::Mouse(SG_MOUSE mouse, GLuint x_pos, GLuint y_pos)
{
	// Forward or backward objects when wheeled
	if (mouse == SG_MOUSE::SG_MOUSE_WHEEL_FORWARD)
	{
#ifdef _DEBUG
		pstatus("mouse wheel forward");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
	if (mouse == SG_MOUSE::SG_MOUSE_WHEEL_BACKWARD)
	{
#ifdef _DEBUG
		pstatus("mouse wheel backward");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}

	// Convert mouse position to OpenGL style
	m_hAct->ConvertMFCPosition(&x_pos, &y_pos);

	// Mouse events
	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN)
	{
#ifdef _DEBUG
		pstatus("mouse left button down");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_DOWN)
	{
#ifdef _DEBUG
		pstatus("mouse right button down");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP)
	{
#ifdef _DEBUG
		pstatus("mouse left button up");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
	if (mouse == SG_MOUSE::SG_MOUSE_R_BUTTON_UP)
	{
#ifdef _DEBUG
		pstatus("mouse right button up");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
	if (mouse == SG_MOUSE::SG_MOUSE_MOVE)
	{
#ifdef _DEBUG
		pstatus("mouse is moving");
#endif
		MouseMotion(mouse, x_pos, y_pos);
	}
};


void Visualization::Keyboard(SG_KEYS keys, SG_KEY_STATUS status)
{ 
	switch ( keys )
	{
	case SG_KEYS::SG_KEY_R:
		m_view->rotate_of_y += 0.3f;
		break;

	case SG_KEYS::SG_KEY_D:
		m_view->rotate_of_x += 0.3f;
		break;

	case SG_KEYS::SG_KEY_Z:
		m_view->rotate_of_x = 0.f;
		m_view->rotate_of_y = 0.f;
		break;
	
	default: break;
	}
}


GLuint Visualization::TEXEL2(GLuint s, GLuint t) { return  BYTES_PER_TEXEL * (s * m_volume->width + t); };


GLuint Visualization::TEXEL3(GLuint s, GLuint t,  GLuint r) { return  TEXEL2(s, t) + LAYER(r); };


GLuint Visualization::LAYER(GLuint r) {return m_volume->width * m_volume->height * r * BYTES_PER_TEXEL;};