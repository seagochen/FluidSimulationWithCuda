/**
* <Author>      Orlando Chen
* <First>       Jan 07, 2014
* <Last>		Jan 08, 2014
* <File>        LaunchMainDynamic.cpp
*/

#include <GL\glew32c.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>
#include <iostream>
#include <memory>

#include "FunctionHelperDynamic.h"
#include "FluidSimAreaDynamic.h"
#include "VolumeRenderingDynamic.h"
#include "resource.h"

sge::fluidsim      m_fluid;
sge::VolumeHelper  m_vh;
sge::FluidSimProc *m_fs;
sge::MainActivity *activity;

void initialize ()
{
	using namespace sge;

	m_fluid.ray.fStepsize     = STRIDE;
	m_fluid.ray.nAngle        = 0;
	m_fluid.ray.uCanvasWidth  = CANVAS_WIDTH;
	m_fluid.ray.uCanvasHeight = CANVAS_HEIGHT;
	m_fluid.ray.bRun          = true;

	m_fluid.volume.uWidth    = Grids_X;
	m_fluid.volume.uHeight   = Grids_X;
	m_fluid.volume.uDepth    = Grids_X;
	m_fluid.area.uWidth      = Area_X;
	m_fluid.area.uHeight     = Area_X;
	m_fluid.area.uDepth      = Area_X;

	/* create big data */
	size_t size = m_fluid.area.uWidth * m_fluid.area.uHeight * m_fluid.area.uDepth;
	m_fluid.area.ptrDens  = new double [ size ];
	m_fluid.area.ptrVelU  = new double [ size ];
	m_fluid.area.ptrVelV  = new double [ size ];
	m_fluid.area.ptrVelW  = new double [ size ];

	/* GLSL source file path */
	m_fluid.shader.szCanvasVert = ".\\shader\\backface.vert";
	m_fluid.shader.szCanvasFrag = ".\\shader\\backface.frag";
	m_fluid.shader.szVolumVert  = ".\\shader\\raycasting.vert";
	m_fluid.shader.szVolumFrag  = ".\\shader\\raycasting.frag";

	m_fs = new FluidSimProc ( &m_fluid );
	std::cout << "initial stage finished" << std::endl;
};

DWORD WINAPI SolveFluidSimulationProc ( LPVOID lpParam )
{
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
		std::cout << "glewInit failed: " << glewGetErrorString (error) << std::endl;
		exit (1);
	}

	/* create sub-thread function */
	m_fluid.thread.hThread = CreateThread ( 
            NULL,                       // default security attributes
            0,                          // use default stack size  
            SolveFluidSimulationProc,   // thread function name
            NULL,                       // argument to thread function 
            0,                          // use default creation flags 
			&m_fluid.thread.dwThreadId);// returns the thread identifier

	 if ( m_fluid.thread.hThread == NULL )
	 {
		 std::cout << "create sub-thread failed" << std::endl;
		 exit (1);
	 }

	/* initialize the shader program and textures */
	m_vh.CreateShaderProg ( &m_fluid );
	m_fluid.textures.hTexture1D   = m_vh.Create1DTransFunc ( m_vh.DefaultTransFunc() );
	m_fluid.textures.hTexture2D   = m_vh.Create2DCanvas ( &m_fluid );
	m_fluid.textures.hTexture3D   = m_vh.Create3DVolumetric ();
	m_fluid.ray.hCluster          = m_vh.InitVerticesBufferObj ();
	m_fluid.textures.hFramebuffer = m_vh.Create2DFrameBuffer ( &m_fluid );

	std::cout << "initialize finished, sge will start soon!" << std::endl;
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

	const char *szTitle = "Excalibur OTL 1.00.00 alpha test  |  FPS: %d  |  dynamic tracking  |";
	SetWindowText (	activity->GetHWND(), string_fmt ( szTitle, m_fluid.fps.uFPS ).c_str() );
}

void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	/* bind the vertex buffer object to shader with attribute "vertices" */
	glBindAttribLocation ( m_fluid.shader.hProgram, 0, "vertices" );

    /* rendering the framebuffer */
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, m_fluid.textures.hFramebuffer );
	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hBFVert, m_fluid.shader.hBFFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	m_vh.RenderingFace ( GL_FRONT, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	/* unbind the framebuffer, do the volume rendering */
    glBindFramebuffer ( GL_FRAMEBUFFER, 0 );
	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hRCVert, m_fluid.shader.hRCFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	m_vh.SetVolumeInfoUinforms ( &m_fluid );
	m_vh.RenderingFace ( GL_BACK, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	/* couting fps */
	CountFPS ();
};

void onDestroy ()
{
	m_fluid.ray.bRun = false;
	WaitForSingleObject ( m_fluid.thread.hThread, INFINITE );
	CloseHandle ( m_fluid.thread.hThread );

	m_fs->FreeResourcePtrs ();
	SAFE_DELT_PTR ( m_fluid.area.ptrDens );
	SAFE_DELT_PTR ( m_fluid.area.ptrVelU );
	SAFE_DELT_PTR ( m_fluid.area.ptrVelV );
	SAFE_DELT_PTR ( m_fluid.area.ptrVelW );
	SAFE_FREE_PTR ( m_fs );
	SAFE_FREE_PTR ( m_fluid.shader.ptrShader );

	std::cout << "memory freed, program exits..." << std::endl;
	exit ( 1 );
};

void onKeyboard ( sge::SG_KEYS keys, sge::SG_KEY_STATUS status )
{
	if ( status == sge::SG_KEY_DOWN )
	{
		switch (keys)
		{
		case sge::SG_KEY_Q:
		case sge::SG_KEY_ESCAPE:
			onDestroy();
			break;
		
		case sge::SG_KEY_C:
			m_fs->ZeroData();
			break;
		
		default:
			break;
		}
	}
};

void onMouse ( sge::SG_MOUSE mouse, GLuint x, GLuint y, int degree )
{
	if ( mouse eqt sge::SG_MOUSE_WHEEL_FORWARD or
		mouse eqt sge::SG_MOUSE_WHEEL_BACKWARD )
	{
		m_fluid.ray.nAngle = ( m_fluid.ray.nAngle + degree ) % 360;
	}
};

int main()
{
	initialize ();

	activity = new sge::MainActivity ( SCREEN_WIDTH, SCREEN_HEIGHT );

	activity->SetAppClientInfo ( IDI_ICON1, IDI_ICON1 );
	activity->RegisterCreateFunc ( onCreate );
	activity->RegisterDisplayFunc ( onDisplay );
	activity->RegisterMouseFunc ( onMouse );
	activity->RegisterDestroyFunc ( onDestroy );
	activity->RegisterKeyboardFunc ( onKeyboard );
	
	activity->SetupRoutine ();
}