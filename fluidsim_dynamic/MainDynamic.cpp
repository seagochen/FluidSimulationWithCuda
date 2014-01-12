/**
* <Author>      Orlando Chen
* <First>       Oct 16, 2013
* <Last>		Jan 12, 2014
* <File>        MainDynamic.cpp
*/

#include <GL\glew32c.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <GLM\glm.hpp>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtx\transform2.hpp>
#include <GLM\gtc\type_ptr.hpp>
#include <iostream>
#include <memory>

#include "FluidSimAreaDynamic.h"
#include "RenderingDynamic.h"
#include "FunctionHelperDynamic.h"
#include "resource.h"

using namespace sge;
using namespace std;

fluidsim      m_fluid;
VolumeHelper  m_vh;
FluidSimProc *m_fs;
MainActivity *activity;
int index;

void initialize ()
{
	index = 0;

	m_fluid.ray.fStepsize     = STEPSIZE;
	m_fluid.ray.nAngle        = 0;
	m_fluid.ray.uCanvasWidth  = Canvas_X;
	m_fluid.ray.uCanvasHeight = Canvas_X;
	m_fluid.ray.bRun          = true;
	m_fluid.volume.uWidth     = Grids_X;
	m_fluid.volume.uHeight    = Grids_X;
	m_fluid.volume.uDepth     = Grids_X;
	m_fluid.area.uWidth       = Grids_X;
	m_fluid.area.uHeight      = Grids_X;
	m_fluid.area.uDepth       = Grids_X;

	m_fluid.shader.szCanvasVert = ".\\shader\\backface.vert";
	m_fluid.shader.szCanvasFrag = ".\\shader\\backface.frag";
	m_fluid.shader.szVolumVert  = ".\\shader\\raycasting.vert";
	m_fluid.shader.szVolumFrag  = ".\\shader\\raycasting.frag";

	/* prepare the fluid simulation stage */
	m_fs = new FluidSimProc ( &m_fluid );

	cout << "initial stage finished" << endl;
};

DWORD WINAPI FluidSimulationProc ( LPVOID lpParam )
{
	/* solve the fluid simulation */
	while ( m_fluid.ray.bRun )
	{
		m_fs->FluidSimSolver ( &m_fluid );
	}

	return 0;
};

void onCreate ()
{
	/* initialize glew */
	GLenum error = glewInit ();
	if ( error != GLEW_OK )
	{
		cout << "glewInit failed: " << glewGetErrorString (error) << endl;
		exit (1);
	}

	/* create sub-thread function */
	m_fluid.thread.hThread = CreateThread ( 
            NULL,                   // default security attributes
            0,                      // use default stack size  
            FluidSimulationProc,    // thread function name
            NULL,                   // argument to thread function 
            0,                      // use default creation flags 
			&m_fluid.thread.dwThreadId);   // returns the thread identifier

	 if ( m_fluid.thread.hThread == NULL )
	 {
		 cout << "create sub-thread failed" << endl;
		 exit (1);
	 }

	/* initialize the shader program and textures */
	m_vh.CreateShaderProg ( &m_fluid );
	m_fluid.textures.hTexture1D   = m_vh.Create1DTransFunc ( m_vh.DefaultTransFunc () );
	m_fluid.textures.hTexture2D   = m_vh.Create2DCanvas ( &m_fluid );
	m_fluid.textures.hTexture3D   = m_vh.Create3DVolumetric ();
	m_fluid.ray.hCluster          = m_vh.InitVerticesBufferObj ();
	m_fluid.textures.hFramebuffer = m_vh.Create2DFrameBuffer ( &m_fluid );

	cout << "initialize finished, sge will start soon!" << endl;
};

void CountFPS()
{
	/* counting FPS */
	m_fluid.fps.dwFrames ++;
	m_fluid.fps.dwCurrentTime = GetTickCount();
	m_fluid.fps.dwElapsedTime = m_fluid.fps.dwCurrentTime - m_fluid.fps.dwLastUpdateTime;

	/* 1 second */
	if ( m_fluid.fps.dwElapsedTime >= 1000 )
	{
		m_fluid.fps.uFPS     = m_fluid.fps.dwFrames * 1000 / m_fluid.fps.dwElapsedTime;
		m_fluid.fps.dwFrames = 0;
		m_fluid.fps.dwLastUpdateTime = m_fluid.fps.dwCurrentTime;
	}

	const char *szTitle = "Excalibur OTL 1.10.00 alpha test  |  FPS: %d  |  dynamic tracking  |";
	SetWindowText (	activity->GetHWND(), string_fmt ( szTitle, m_fluid.fps.uFPS ).c_str() );
}

void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	/* bind the vertex buffer object to shader with attribute "vertices" */
	glBindAttribLocation ( m_fluid.shader.hProgram, 0, "vertices" );

    /* do Render Now! */
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, m_fluid.textures.hFramebuffer );
	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hBFVert, m_fluid.shader.hBFFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	m_vh.RenderingFace ( GL_FRONT, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	/* do not bind the framebuffer now */
    glBindFramebuffer ( GL_FRAMEBUFFER, 0 );

	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hRCVert, m_fluid.shader.hRCFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	m_vh.SetVolumeInfoUinforms ( &m_fluid );
	m_vh.RenderingFace ( GL_BACK, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	CountFPS ();
};

void onDestroy ()
{
	m_fluid.ray.bRun = false;
	WaitForSingleObject ( m_fluid.thread.hThread, INFINITE );
	CloseHandle ( m_fluid.thread.hThread );

	m_fs->FreeResourcePtrs();  
	SAFE_FREE_PTR ( m_fs );
	SAFE_FREE_PTR ( m_fluid.shader.ptrShader );

	cout << "memory freed, program exits..." << endl;
	exit(1);
};

void onKeyboard ( SG_KEYS keys, SG_KEY_STATUS status )
{
	if ( status == SG_KEY_STATUS::SG_KEY_DOWN )
	{
		switch (keys)
		{
		case sge::SG_KEY_Q:
		case sge::SG_KEY_ESCAPE:
			onDestroy ();
			break;

		case sge::SG_KEY_T:
			index = (index + 1) % 8;
			m_fs->SelectNode ( index );
			break;
		
		case sge::SG_KEY_C:
			m_fs->ZeroData ();
			break;
		
		default:
			break;
		}
	}
};

void onMouse ( SG_MOUSE mouse, uint x, uint y, int degree )
{
	if ( mouse eqt SG_MOUSE_WHEEL_FORWARD or mouse eqt SG_MOUSE_WHEEL_BACKWARD )
	{
		m_fluid.ray.nAngle = (m_fluid.ray.nAngle + degree) % 360;
	}
};

int main()
{
	initialize ();

	activity = new MainActivity ( Windows_X, Windows_X );

	activity->SetAppClientInfo ( IDI_ICON1, IDI_ICON1 );
	activity->RegisterCreateFunc ( onCreate );
	activity->RegisterDisplayFunc ( onDisplay );
	activity->RegisterMouseFunc ( onMouse );
	activity->RegisterDestroyFunc ( onDestroy );
	activity->RegisterKeyboardFunc ( onKeyboard );
	
	activity->SetupRoutine();
}