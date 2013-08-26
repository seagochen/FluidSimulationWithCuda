// 导入自定义头文件
#include "Header.h"
#include "Visualization.h"


/* 获取有关OpenGL函数扩展地址，这些操作只需要在Windows平台中 */
int seago::Visualization::loadFunc()
{
	if ( (glTexImage3D = (PFNGLTEXIMAGE3DPROC) wglGetProcAddress("glTexImage3D")) == NULL )
	{
		printf("Error in line %d: Couldn't load glTexImage3D function. Aborting.\n", __LINE__);
		return FAIL;
	}
	if ( (glTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC)wglGetProcAddress("glTexSubImage3D")) == NULL )
	{
		printf("Error in line %d: Couldn't load glTexSubImage3D function. Aborting.\n", __LINE__);
		return FAIL;
	}

	return OK;
};


/* Create and bind 3-D texture */
int seago::Visualization::bindTexutre()
{
	// 创建3D纹理
	glGenTextures(1, &m_volume.textureID);
	glBindTexture(GL_TEXTURE_3D, m_volume.textureID);

	// 设定贴图参数
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); // scale nearest when image bigger than texture
	glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // scale nearest when image smaller than texture

	// 实现3D纹理相关函数
	if (loadFunc() == FAIL) {
		printf("Error in line %d: Couldn't load OpenGL 3-D functions. Aborting.\n", __LINE__);
		return FAIL;
	}

	// 载入3D纹理，并设置相关参数
	glTexImage3D(GL_TEXTURE_3D, // GLenum target
		0,		       // GLint level,
		GL_RGB,        // GLint internalFormat
		m_volume.width,  // GLsizei width
		m_volume.height, // GLsizei height
		m_volume.depth,  // GLsizei depth
		0,             // GLint border
		GL_RGB,        // GLenum format
		GL_UNSIGNED_BYTE, // GLenum type
		m_volume.data);  // const GLvoid * data

	return OK;
};


void seago::Visualization::initFPS()
{
	// Zero out the frames per second variables:
	m_fps.dwFrames = 0;
	m_fps.dwCurrentTime = 0;
	m_fps.dwLastUpdateTime = 0;
	m_fps.dwElapsedTime = 0;
};


/* An OpenGL initialization function */
void seago::Visualization::sgInit(int width, int height)
{
	// Load the textures
	if ( bindTexutre() == FAIL ) {
		printf("Cannot initialize the program, exit...\n");
		exit(0);
	}
	// Set clearing color
	glClearColor(0.f, 0.f, 0.7f, 1.f);
	// Enable clearing of the depth buffer
	glClearDepth(1.f);
	// Type of depth test to do
	glDepthFunc(GL_LESS);
	// Enable depth testing
	glEnable(GL_DEPTH_TEST);
	// Enable smooth color shading
	glShadeModel(GL_SMOOTH);
	// Changing matrix
	glMatrixMode(GL_PROJECTION);
	// Reset the projection matrix
	glLoadIdentity();
	// Calculate the aspect ratio of the window
	gluPerspective(m_window.field_of_view_angle,
		(GLfloat)width / (GLfloat)height, m_window.z_near, m_window.z_far);
	// Changing matrix 
	glMatrixMode(GL_MODELVIEW);	

	// etc...
	initFPS();
	initViewMatrix();
}


/* The function called when window is resized */
void seago::Visualization::sgResizeScreen(int width, int height)
{
	// Prevent a divide by zero if the window is too small
	if (height == 0) height = 1;

	glViewport(0, 0, width, height);

	// Reset the current viewport and perspective transformation
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(m_window.field_of_view_angle,
		(GLfloat)width / (GLfloat)height, m_window.z_near, m_window.z_far);
	glMatrixMode(GL_MODELVIEW);
}


/* Display CALLBACK function */
void seago::Visualization::sgDisplay() 
{
	// 将数据实时显示在屏幕上，这里采用glTexSubImage3D而不是glTexImage3D，是为了降低频繁更新而带来的性能损耗
	glTexSubImage3D(GL_TEXTURE_3D, // GLenum target,
		0, // GLint level,
		0, // GLint xoffset,
		0, // GLint yoffset,
		0, // GLint zoffset,
		m_volume.width,  // GLsizei width,
		m_volume.height, // GLsizei height,
		m_volume.depth,  // GLsizei depth,
		GL_RGB,        // GLenum format,
		GL_UNSIGNED_BYTE, // GLenum type,
		m_volume.data);  // const GLvoid * data

	// Clear Screen and Depth Buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// Reset the view
	glLoadIdentity();

	// 移动代理几何
	glTranslatef(0.0f, 0.0f, m_view.nearZ);
	
	// 通过移动Camera实现对模型的观察
	gluLookAt(
		m_view.eyeX, m_view.eyeY, m_view.eyeZ, // eye
		0, 0, 0, // center
		0, 1, 0);// Up

	// 绘制代理几何
	glEnable(GL_TEXTURE_3D);
	agentBox();
	glDisable(GL_TEXTURE_3D);

	// Update FPS
	sgCountFPS();

	// swap buffer and display image
	//glutSwapBuffers();
}


/* Draw an agent geometry for 3-D texture */
void seago::Visualization::agentBox()
{
	// Draw 6 quadrilaterals
	glBegin(GL_QUADS);

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

	glEnd();
};


/* FPS function */
void seago::Visualization::sgCountFPS( ) {

	std::stringstream buffer;
	
	// Calculate the number of frames per one second:
	m_fps.dwFrames++;
	m_fps.dwCurrentTime = GetTickCount(); // Even better to use timeGetTime()
	m_fps.dwElapsedTime = m_fps.dwCurrentTime - m_fps.dwLastUpdateTime;
	
	if( m_fps.dwElapsedTime >= 1000 )
	{
		UINT FPS = m_fps.dwFrames * 1000.0 / m_fps.dwElapsedTime;
		system("cls");
		std::cout<<FPS<<std::endl;
		//buffer<<" FPS: "<<std::to_string(FPS);
		//std::cout<<buffer.str().c_str()<<std::endl;
		//glutSetWindowTitle( buffer.str().c_str() );
		m_fps.dwFrames = 0;
		m_fps.dwLastUpdateTime = m_fps.dwCurrentTime;
	}
}


int seago::Visualization::sgLoadVolumeData(seago::Visualization::_volumeData const *data_in)
{
	m_volume.data = data_in->data;
	m_volume.width = data_in->width;
	m_volume.height = data_in->height;
	m_volume.depth = data_in->depth;

	if ( m_volume.data != NULL ) return OK;
	
	return FAIL;
};


// 设置窗口参数
void seago::Visualization::sgSetWindowParam(seago::Visualization::_windowSets const *win_in)
{
	m_window.field_of_view_angle = win_in->field_of_view_angle;
	m_window.title = win_in->title;
	m_window.z_far = win_in->z_far;
	m_window.z_near = win_in->z_near;
	m_window.width = win_in->width;
	m_window.height = win_in->height;
};



void seago::Visualization::sgMotion(int x,int y)
{
	if(m_mouse.isLeftHold)
	{
		float xDis = x - m_mouse.preX;
		float yDis = y - m_mouse.preY;
		m_mouse.preX = x;
		m_mouse.preY = y;
		
		m_view.rotateX += xDis;
		m_view.rotateY += yDis;

		// 计算摄像机空间坐标
		m_view.eyeX = cos ( m_view.rotateX / 15.f ) * m_view.radius;
		m_view.eyeZ = sin ( m_view.rotateX / 15.f ) * m_view.radius;
		m_view.eyeY = m_view.rotateY / 15.f;

		if ( m_view.eyeY > m_view.radius ) m_view.eyeY = m_view.radius;
		if ( m_view.eyeY <-m_view.radius ) m_view.eyeY =-m_view.radius;

		//glutPostRedisplay();
	}
};



void seago::Visualization::sgMouse(int button,int state,int x,int y)
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
	// 检查鼠标左右键的状态
//	 if (state == GLUT_DOWN)
//	 {
//		 m_mouse.preX = x;
//		 m_mouse.preY = y;
//		 
//		 if (button==GLUT_LEFT_BUTTON) m_mouse.isLeftHold = true;
//		 else if (button==GLUT_RIGHT_BUTTON) m_mouse.isRightHold = true;
//	 }
//	 else if (state == GLUT_UP)
//	 {
//		 if (button == GLUT_LEFT_BUTTON) m_mouse.isLeftHold = false;
//		 else if (button==GLUT_RIGHT_BUTTON) m_mouse.isRightHold = false;
//	 }
};


void seago::Visualization::initViewMatrix(void)
{
	m_view.eyeX = 0.0f;
	m_view.eyeY = 0.0f;
	m_view.eyeZ = 2.0f;
	m_view.radius = 2.0f;
	m_view.nearZ = -5.f;
};


/* Keyboard CALLBACK function */
//void seago::Visualization::sgKeyboard ( unsigned char key, int mousePositionX, int mousePositionY )
void seago::Visualization::sgKeyboard( sge::SGKEYS keys)
{ 
	using namespace sge;

	switch ( keys )
	{
		// ESC default
	
	default: break;
	}
}