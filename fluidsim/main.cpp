#include <GL\glew32c.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <GLM\glm.hpp>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtx\transform2.hpp>
#include <GLM\gtc\type_ptr.hpp>
#include <iostream>

#include "fluidsim.h"
#include "volumeHelper.h"

using namespace sge;
using namespace std;

fluidsim     m_fluid;
VolumeHelper m_vh;
FluidSim    *m_fs;

#define K_ON 100

void initialize ()
{
	m_fluid.fStepsize = 0.001f;
	m_fluid.nAngle = 0;
	m_fluid.nCanvasWidth  = 600;
	m_fluid.nCanvasHeight = 600;
#if K_ON
	m_fluid.nVolWidth    = param::nGrids_X;
	m_fluid.nVolHeight   = param::nGrids_X;
	m_fluid.nVolDepth    = param::nGrids_X;
#else
	m_fluid.nVolWidth    = 256;
	m_fluid.nVolHeight   = 256;
	m_fluid.nVolDepth    = 225;
#endif
	m_fluid.szCanvasVert = ".\\shader\\backface.vert";
	m_fluid.szCanvasFrag = ".\\shader\\backface.frag";
	m_fluid.szVolumVert  = ".\\shader\\raycasting.vert";
	m_fluid.szVolumFrag  = ".\\shader\\raycasting.frag";

	m_fluid.ptrData = (GLubyte*) calloc (param::nSim_Size, sizeof(GLubyte));

#if !K_ON
	m_vh.LoadVolumeSource ( ".\\res\\head256.raw", &m_fluid );
#endif

	/// Prepare the fluid simulation stage ///
	m_fs = new FluidSim ( &m_fluid );

	cout << "initial stage finished" << endl;
};


DWORD WINAPI cudaCFD ( LPVOID lpParam )
{
#if K_ON
	/// Solve the fluid simulation ///
	m_fs->FluidSimSolver ( &m_fluid );
#endif

	return 0;
};


#pragma region callback functions

void onCreate ()
{
	/// Initialize glew ///
	GLenum error = glewInit ();
	if ( error != GLEW_OK )
	{
		cout << "glewInit failed: " << glewGetErrorString (error) << endl;
		exit (1);
	}

	/// Create sub-thread function ///
	m_fluid.hThread = CreateThread ( 
            NULL,                   // default security attributes
            0,                      // use default stack size  
            cudaCFD,                // thread function name
            NULL,                   // argument to thread function 
            0,                      // use default creation flags 
			&m_fluid.dwThreadId);   // returns the thread identifier

	 if ( m_fluid.hThread == NULL )
	 {
		 cout << "create sub-thread failed" << endl;
		 exit (1);
	 }

	/// Initialize the shader program and textures ///
	m_vh.CreateShaderProg ( &m_fluid );
	m_fluid.hTexture1D = m_vh.Create1DTransFunc ( m_vh.DefaultTransFunc () );
	m_fluid.hTexture2D = m_vh.Create2DCanvas ( &m_fluid );
	m_fluid.hTexture3D = m_vh.Create3DVolumetric ();
	m_fluid.hCluster = m_vh.InitVerticesBufferObj ();
	m_fluid.hFramebuffer = m_vh.Create2DFrameBuffer ( &m_fluid );

	cout << "initialize finished, sge will start soon!" << endl;
};

void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	/// Bind the vertex buffer object to shader with attribute "vertices" ///
	glBindAttribLocation ( m_fluid.hProgram, 0, "vertices" );

    /// Do Render Now! ///
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, m_fluid.hFramebuffer );
	glViewport ( 0, 0, m_fluid.nCanvasWidth, m_fluid.nCanvasHeight );
	m_fluid.ptrShader->LinkShaders ( m_fluid.hProgram, 2, m_fluid.hBFVert, m_fluid.hBFFrag );
	m_fluid.ptrShader->ActiveProgram ( m_fluid.hProgram );
	m_vh.RenderingFace ( GL_FRONT, &m_fluid );
	m_fluid.ptrShader->DeactiveProgram ( m_fluid.hProgram );

	/// Do not bind the framebuffer now ///
    glBindFramebuffer ( GL_FRAMEBUFFER, 0 );

	glViewport ( 0, 0, m_fluid.nCanvasWidth, m_fluid.nCanvasHeight );
	m_fluid.ptrShader->LinkShaders ( m_fluid.hProgram, 2, m_fluid.hRCVert, m_fluid.hRCFrag );
	m_fluid.ptrShader->ActiveProgram ( m_fluid.hProgram );
	m_vh.SetVolumeInfoUinforms ( &m_fluid );
	m_vh.RenderingFace ( GL_BACK, &m_fluid );
	m_fluid.ptrShader->DeactiveProgram ( m_fluid.hProgram );

	m_fluid.nAngle = (m_fluid.nAngle + 1) % 360;
};

void onDestroy ()
{
	SAFE_FREE_PTR ( m_fluid.ptrData );
	SAFE_FREE_PTR ( m_fluid.ptrShader );

	CloseHandle ( m_fluid.hThread );

	cout << "memory freed, program exits..." << endl;
};

void onKeyboard ( SG_KEYS keys, SG_KEY_STATUS status )
{
	if ( keys == SG_KEYS::SG_KEY_ESCAPE && status == SG_KEY_STATUS::SG_KEY_DOWN )
	{
		void onDestroy ();
		exit (1);
	}
};

#pragma endregion


int main()
{
	initialize ();

	MainActivity *activity = new MainActivity ( m_fluid.nCanvasWidth, m_fluid.nCanvasHeight );

	activity->SetAppClientInfo ( L"Excalibur OTL 0.00.02.01" );

	activity->RegisterCreateFunc ( onCreate );
	activity->RegisterDisplayFunc ( onDisplay );
	activity->RegisterDestroyFunc ( onDestroy );
	activity->RegisterKeyboardFunc ( onKeyboard );
	
	activity->SetupRoutine();
}