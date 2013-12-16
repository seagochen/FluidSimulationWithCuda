#define INCLUDE_GLEW32_CODE

#include "main.h"

GLuint prog, bfVert, bfFrag, rcVert, rcFrag;
GLuint cluster;
GLuint tex1D, tex2D, tex3D;
GLuint framebuffer;
Shader *shader;
GLint angle = 0.f;

void initRuntime ()
{
	GLenum error = glewInit ();
	if ( error != GLEW_OK )
	{
		cout << "glewInit failed: " << glewGetErrorString (error) << endl;
		exit (1);
	}
}


void onCreate ()
{
	// Initialize glew
	initRuntime ();

	// Create ...
	CreateShaders ( shader, &prog, &bfVert, &bfFrag, &rcVert, &rcFrag );

	tex1D = Create1DTransFunc ();
	tex2D = Create2DBackFace ();
	tex3D = Create3DVolumetric ();

	cluster = InitVerticesBufferObj ();

	framebuffer = CreateFrameBuffer ( tex2D );

	// Specify the background clearing color
	glClearColor(0.7f, 0.7f, 0.7f, 0.5f);

	cout << "initialize finished, sge will work soon!" << endl;
};



void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	// Bind index 0 to the shader input variable "VerPos"
	glBindAttribLocation ( prog, 0, "VerPos" );
	// Bind index 1 to the shader input variable "VerClr"
	glBindAttribLocation ( prog, 1, "VerClr" );

    /// Do Render Now!
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, framebuffer ); // Chose which framebuffer to render
	glViewport ( 0, 0, BACKFACE_SIZE_X, BACKFACE_SIZE_X );
	shader->LinkShaders ( prog, 2, bfVert, bfFrag );
	shader->ActiveProgram ( prog );
	//void RenderingFace ( GLenum cullFace, GLfloat angle, GLuint program, GLuint cluster )
	RenderingFace ( GL_FRONT, angle, prog, cluster );   // From front face
	shader->DeactiveProgram ( prog );


    glBindFramebuffer ( GL_FRAMEBUFFER, 0 ); // To break the binding
    glViewport ( 0, 0, BACKFACE_SIZE_X, BACKFACE_SIZE_X );
	shader->LinkShaders ( prog, 2, rcVert, rcFrag );
	shader->ActiveProgram ( prog );
	SetVolumeInfoUinforms ( prog, tex1D, tex2D, tex3D,
		BACKFACE_SIZE_X, BACKFACE_SIZE_X, 0.001f );
    RenderingFace ( GL_BACK, angle, prog, cluster );
	shader->DeactiveProgram ( prog );

	angle = (angle + 1) % 360;
};


int main()
{
	using namespace sge;

	MainActivity *activity = new MainActivity(800, 600);

	activity->SetAppClientInfo(L"Excalibur OTL 0.00.00.001");

	activity->RegisterCreateFunc(onCreate);
	activity->RegisterDisplayFunc(onDisplay);
	
	activity->SetupRoutine();
}