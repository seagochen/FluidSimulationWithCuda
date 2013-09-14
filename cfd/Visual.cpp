#include "Visual.h"

using namespace sge;

static _mouse        *m_mouse;
static _fps          *m_fps;
static _volume2D     *m_volume2D;
static _volume3D     *m_volume3D;
static _viewMatrix   *m_view;
static FreeType      *m_font;
static MainActivity  *m_hAct;
static GLfloat        m_width, m_height;


Visual::Visual(GLuint width, GLuint height, MainActivity *hActivity)
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

	m_volume2D->size = 0;
	m_volume3D->size = 0;
};


Visual::~Visual(void)
{
	OnDestroy();
};


void InitFPS(void)
{
	// Zero out the frames per second variables:
	m_fps->dwFrames = 0;
	m_fps->dwCurrentTime = 0;
	m_fps->dwLastUpdateTime = 0;
	m_fps->dwElapsedTime = 0;
};


void InitFont(void)
{
	m_font->Init("EHSMB.TTF", 12);
}


void InitViewMatrix(void)
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


void InitMouseStatus(void)
{
	m_mouse->left_button_pressed = false;
	m_mouse->right_button_pressed = false;
};


void Setup(void)
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


void SetTexture(void)
{
	// Create 2D image texture and assign an ID
	glGenTextures(1, &m_volume2D->texture_id);

	// Create 3D image texture and assign an ID
	glGenTextures(1, &m_volume3D->texture_id);

	// Check texture ID is available
	if (!m_font->IsTextureIDAvailable(m_volume2D->texture_id))
	{
		ErrorMSG("Cann't assign an available texture ID");
		exit(0);
	}
};


void Visual::OnCreate()
{
	// Initialize
	InitViewMatrix();
	InitFont();
	InitFPS();
	InitMouseStatus();

	// Initialize glew
	glewInit();

	// Call for OpenGL envrionment setup
	Setup();

	// Set texture
	SetTexture();
};


void Visual::OnResize(GLuint width, GLuint height)
{
	// Prevent a divide by zero if the window is too small
	if (height == 0) height = 1;

	m_width  = width;
	m_height = height;

	glViewport(0, 0, width, height);

	// Reset the current viewport and perspective transformation
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(m_view->view_angle, m_width / m_height, m_view->z_near, m_view->z_far);
	glMatrixMode(GL_MODELVIEW);
};


void DrawAgent2D()
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


void CountFPS( ) 
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
		glColor3f(1.0f, 1.0f, 1.0f);
		m_font->EnableFreeType();
		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", m_fps->FPS);
		m_font->DisableFreeType();
	}
	glPopMatrix();
}


void Visual::OnDisplay()
{
	// Reset matrix
	glLoadIdentity();

	// Clear Screen and Depth Buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Set camera
	gluLookAt(
		m_view->eye_x,  m_view->eye_y,  m_view->eye_z,  // eye
		m_view->look_x, m_view->look_y, m_view->look_z, // center
		m_view->up_x,   m_view->up_y,   m_view->up_z);  // Up

	// Draw fluid sim result on 2-D map
	DrawAgent2D();

	// Print FPS
	CountFPS();
};


void Visual::OnKeyboard(SG_KEYS keys, SG_KEY_STATUS status)
{
	if (keys == SG_KEYS::SG_KEY_ESCAPE && status == SG_KEY_STATUS::SG_KEY_DOWN)	
	{
		OnDestroy();
		exit(0);
	}
};


// TODO
void Visual::OnMouse(SG_MOUSE mouse, GLuint x_pos, GLuint y_pos)
{
};


void Visual::OnDestroy()
{
	SAFE_DELT_PTR(m_mouse);
	SAFE_DELT_PTR(m_fps);
	SAFE_DELT_PTR(m_view);

	if (m_volume2D->size > 0) SAFE_FREE_PTR(m_volume2D->data);
	if (m_volume3D->size > 0) SAFE_FREE_PTR(m_volume3D->data);

	if (m_font != NULL)	m_font->Clean();
	SAFE_DELT_PTR(m_font);

#ifdef PRINT_STATUS
	printf("Call OnDestroy, now resource released up!\n");
#endif
};


void Visual::UploadVolumeData(_volume2D const *data_in)
{
	m_volume2D->width  = data_in->width;
	m_volume2D->height = data_in->height;
	m_volume2D->data   = data_in->data;

	m_volume2D->size   = data_in->size;

#ifdef PRINT_STATUS
	system("cls");
	printf("Upload volume data and try to rendering the result, size: %d\n", m_volume2D->size);
#endif
};


void Visual::UploadVolumeData(_volume3D const *data_in)
{
	m_volume3D->width  = data_in->width;
	m_volume3D->height = data_in->height;
	m_volume3D->depth  = data_in->depth;
	m_volume3D->data   = data_in->data;

	m_volume3D->size   = data_in->size;

#ifdef PRINT_STATUS
	system("cls");
	printf("Upload volume data and try to rendering the result, size: %d\n", m_volume3D->size);
#endif
};


int Visual::Texel2D(int i, int j)
{
	return  BYTES_PER_TEXEL * (i * m_volume2D->height + j);
};


int Layer(int layer)
{
	return layer * m_volume3D->width * m_volume3D->height;
};


int Visual::Texel3D(int i, int j, int k)
{
	return BYTES_PER_TEXEL * (Layer(i) + m_volume3D->height * j + k);
};