#include "Visualization.h"

using namespace sge;


#ifndef _SGE_DEFINED_VALUE_
#define _SGE_DEFINED_VALUE_

static _mouse      *m_mouse;
static _fps        *m_fps;
static _volumeData *m_volume;
static _viewMatrix *m_view;
static FreeType    *m_font;

static UINT FPS = 0;
#endif


Visualization::Visualization()
{
	m_mouse  = new _mouse;
	m_fps    = new _fps;
	m_volume = new _volumeData;
	m_view   = new _viewMatrix;
	m_font   = new FreeType;
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
	m_view->field_of_view_angle    = 45.f;
	m_view->radius_of_view_matrix  = 2.f;
	m_view->eye_at_x               = 0.f;
	m_view->eye_at_y               = 0.f;
	m_view->eye_at_z               = 5.f;
	m_view->z_far                  = 100.f;
	m_view->z_near                 = 0.1f;
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


void Visualization::Init(GLuint width, GLuint height)
{
	// Initialize
	glewInit();
	InitFont();
	InitFPS();
	InitViewMatrix();

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

	// Enable depth testing
	glEnable(GL_DEPTH_TEST);

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
	// Clear Screen and Depth Buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// 将数据实时显示在屏幕上，这里采用glTexSubImage3D而不是glTexImage3D，是为了降低频繁更新而带来的性能损耗
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
	
	// 通过移动Camera实现对模型的观察
	gluLookAt(
		m_view->eye_at_x, m_view->eye_at_y, m_view->eye_at_z, // eye
		0, 0, 0,                                           // center
		0, 1, 0);                                             // Up

	// 放置代理幾何
	glPushMatrix();
	{
		// Reset the view
		glLoadIdentity();
		
		// 移动代理几何
		glTranslatef( m_view->rotate_of_x, m_view->rotate_of_y, m_view->rotate_of_z );
		
		// 根據鼠標動作及鍵盤操作旋轉或縮放代理幾何，或者其他操作
		Motion();

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
	
	
	if ( m_fps->dwElapsedTime >= 1000 )
	{
		FPS = m_fps->dwFrames * 1000.0 / m_fps->dwElapsedTime;
		m_fps->dwFrames = 0;
		m_fps->dwLastUpdateTime = m_fps->dwCurrentTime;
	}

	glPushMatrix();
	{
		glLoadIdentity();									// Reset The Current Modelview Matrix
		glTranslatef(0.0f,0.0f,-1.0f);						// Move One Unit Into The Screen
		
		// White Text
		glColor3f(1.0f, 1.0f, 1.0f);
		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", FPS);			
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


void Visualization::Motion()
{
//	if (m_mouse.isLeftHold)
//	{
//		float xDis = x - m_mouse.preX;
//		float yDis = y - m_mouse.preY;
//		m_mouse.preX = x;
//		m_mouse.preY = y;
		
//		m_view.rotateX += xDis;
//		m_view.rotateY += yDis;

		// 计算摄像机空间坐标
//		m_view.eyeX = cos ( m_view.rotateX / 15.f ) * m_view.radius;
//		m_view.eyeZ = sin ( m_view.rotateX / 15.f ) * m_view.radius;
//		m_view.eyeY = m_view.rotateY / 15.f;

//		if ( m_view.eyeY > m_view.radius ) m_view.eyeY = m_view.radius;
//		if ( m_view.eyeY <-m_view.radius ) m_view.eyeY =-m_view.radius;

//	}
};


void Visualization::Mouse(SG_MOUSE mouse, GLuint x_pos, GLuint y_pos)
{
	// 检查鼠标滚轮的状态
//	if (state == GLUT_UP && button == MOUSE_WHEEL_UP)
//	{
//		m_view.nearZ += 0.1f;
//		glutPostRedisplay();
//	}
//	if (state == GLUT_UP && button == MOUSE_WHEEL_DOWN)
//	{
//		m_view.nearZ -= 0.1f;
//		glutPostRedisplay();
//	}

	// 检查鼠标的状态
	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_DOWN)
	{
		m_mouse->pre_cursor_x = x_pos;
		m_mouse->pre_cursor_y = y_pos;
		m_mouse->left_button_pressed = true;
	}
	if (mouse == SG_MOUSE::SG_MOUSE_L_BUTTON_UP)
	{
		m_mouse->left_button_pressed = false;
	}
	if (mouse == SG_MOUSE::SG_MOUSE_MOVE && m_mouse->left_button_pressed)
	{
		m_mouse->cur_cursor_x = x_pos - m_mouse->pre_cursor_x;
		m_mouse->cur_cursor_y = y_pos - m_mouse->pre_cursor_y;
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