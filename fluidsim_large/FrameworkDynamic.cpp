/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Oct 16, 2013
* <Last Time>     Mar 05, 2014
* <File Name>     FrameworkDynamic.cpp
*/

#pragma region framework basic functions

#include <GLM\glm.hpp>
#include <GLM\gtc\matrix_transform.hpp>
#include <GLM\gtx\transform2.hpp>
#include <GLM\gtc\type_ptr.hpp>
#include <iostream>
#include "FrameworkDynamic.h"
#include "FluidSimProc.h"

using namespace sge;
using namespace glm;

using std::cout;
using std::endl;

/* 检查着色器句柄是否有错误  */
bool Framework_v1_0::CheckHandleError ( int nShaderObjs, ... )
{
	/* 当创建的着色器小于1个时，报错 */
	if ( nShaderObjs < 1 )
	{
		cout << "parameters error, check your code..." << endl;
		return false;
	}
	
	/* 对创建的着色器进行检查，若发现有错误，报错 */
	va_list list; int i = 1; bool fin = true;
	va_start ( list, nShaderObjs );
	{
		for ( ; i <= nShaderObjs; i++ )
		{
			GLuint value = va_arg ( list, GLuint );

			if ( value eqt 0 ) // 发现当前句柄存在错误，报错
			{
				cout << "Error> the No." << i << " handle is null" << endl;
				fin = false;
			}
		}
	}
	va_end ( list );

	return fin;
};


/* 创建着色器句柄及着色程序 */
void Framework_v1_0::CreateShaderProg ( FLUIDSPARAM *fluid )
{
	/* 创建指针指向着色器及着色器程序 */
	SGHANDLER *prog_out   = &fluid->shader.hProgram;
	SGHANDLER *bfVert_out = &fluid->shader.hBFVert;
	SGHANDLER *bfFrag_out = &fluid->shader.hBFFrag;
	SGHANDLER *rcVert_out = &fluid->shader.hRCVert;
	SGHANDLER *rcFrag_out = &fluid->shader.hRCFrag;

	/* 创建着色器程序 */
	SGSHADER *shader_out = new SGSHADER();

	/* 使用着色器程序读取GLSL代码，并创建着色器 */
	shader_out->CreateShaderObj ( fluid->shader.szCanvasVert, SG_VERTEX,   bfVert_out );
	shader_out->CreateShaderObj ( fluid->shader.szCanvasFrag, SG_FRAGMENT, bfFrag_out );
	shader_out->CreateShaderObj ( fluid->shader.szVolumVert,  SG_VERTEX,   rcVert_out );
	shader_out->CreateShaderObj ( fluid->shader.szVolumFrag,  SG_FRAGMENT, rcFrag_out );

	/* Check error */
	if ( !CheckHandleError ( 4, *bfVert_out, *bfFrag_out, *rcVert_out, *rcFrag_out ) )
	{
		cout << "create shaders object failed" << endl;
		exit (1);
	}
	
	/* Create shader program object */
	shader_out->CreateProgmObj ( prog_out );

	/* Check error */
	if ( !CheckHandleError ( 1, *prog_out) )
	{
		cout << "create program object failed" << endl;
		exit (1);
	}

	/* 链接已创建的着色器程序，并打印信息 */
	fluid->shader.ptrShader = shader_out;
	cout << "shader program created" << endl;
}


/* 创建默认的传递函数，该函数将对3D体数据进行渲染，着色 */
GLubyte* Framework_v1_0::DefaultTransFunc ()
{
	/* Hardcode the transfer function */
	GLubyte *tff = (GLubyte *) calloc ( TPBUFFER_X, sizeof(GLubyte) );
	for ( int i = 0; i < 256; i++ )
	{
		if ( i > 0 )
		{
			tff [ i * 4 + 0 ] = i;
			tff [ i * 4 + 1 ] = (i) % 100 + 40;
			tff [ i * 4 + 2 ] = (i) % 55 + 30;
			tff [ i * 4 + 3 ] = 7;
		}
	}

	/* 创建完毕，打印信息 */
	cout << "use default transfer function" << endl;
	return tff;
}


/* 创建1D纹理信息 */
SGHANDLER Framework_v1_0::Create1DTransFunc ( GLubyte *transfer )
{
	SGHANDLER tff1DTex;

    glGenTextures(1, &tff1DTex);
    glBindTexture(GL_TEXTURE_1D, tff1DTex);

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA8, 256, 0, GL_RGBA, GL_UNSIGNED_BYTE, transfer);
    
	SAFE_FREE_PTR (transfer);

	/* 创建完毕，返回1D纹理句柄并打印信息 */
	cout << "transfer function created" << endl;
	return tff1DTex;
};


/* 创建2D canvas 纹理信息 */
SGHANDLER Framework_v1_0::Create2DCanvas ( FLUIDSPARAM *fluid )
{
    SGHANDLER backFace2DTex;

    glGenTextures(1, &backFace2DTex);
    glBindTexture(GL_TEXTURE_2D, backFace2DTex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 
		fluid->ray.uCanvasWidth, fluid->ray.uCanvasHeight, 0, GL_RGBA, GL_FLOAT, NULL);

	/* 创建完毕，返回2D纹理句柄并打印信息 */
	cout << "canvas created" << endl;
	return backFace2DTex;
};


/* 创建3D纹理信息 */
SGHANDLER Framework_v1_0::Create3DVolumetric ( void )
{
	/* Generate 3D textuer */
	SGHANDLER volTex;
    glGenTextures(1, &volTex);
    glBindTexture(GL_TEXTURE_3D, volTex);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

	/* 创建完毕，返回3D纹理句柄并打印信息 */
	cout << "volumetric texture created" << endl;
    return volTex;
};


/* 创建2D framebuffer 纹理信息 */
GLuint Framework_v1_0::Create2DFrameBuffer ( FLUIDSPARAM *fluid )
{
    /* Create a depth buffer for framebuffer */
    GLuint depthBuffer;
    glGenRenderbuffers(1, &depthBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 
		fluid->ray.uCanvasWidth, fluid->ray.uCanvasHeight);

    /* Attach the texture and the depth buffer to the framebuffer */
	GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 
		fluid->textures.hTexture2D, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	
	/* Check Framebuffer status */
	if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE )
    {
		cout << "framebuffer is not complete" << endl;
		exit(EXIT_FAILURE);
    }
    glEnable(GL_DEPTH_TEST);    

	cout << "framebuffer created" << endl;
	
	return framebuffer;
};


/* 创建代理几何的句柄 */
SGHANDLER Framework_v1_0::CreateVerticesBufferObj ( void )
{
	// How agent cube looks like by specified the coordinate positions of vertices
	GLfloat vertices[24] = 
	{                    // (x, y, z)
		0.0, 0.0, 0.0,   // (0, 0, 0)
		0.0, 0.0, 1.0,   // (0, 0, 1)
		0.0, 1.0, 0.0,   // (0, 1, 0)
		0.0, 1.0, 1.0,   // (0, 1, 1)
		1.0, 0.0, 0.0,   // (1, 0, 0)
		1.0, 0.0, 1.0,   // (1, 0, 1)
		1.0, 1.0, 0.0,   // (1, 1, 0)
		1.0, 1.0, 1.0    // (1, 1, 1)
	};

	// Drawing six faces of agent cube with triangles by counter clockwise
	GLuint indices[36] = 
	{
		/// <front> 1 5 7 3 </front>///
		1,5,7,
		7,3,1,
		/// <back> 0 2 6 4 </back> ///
		0,2,6,
		6,4,0,
		/// <left> 0 1 3 2 </left> ///
		0,1,3,
		3,2,0,
		/// <right> 7 5 4 6 </right> ///
		7,5,4,
		4,6,7,
		/// <up> 2 3 7 6 </up> ///
		2,3,7,
		7,6,2,
		/// <down> 1 0 4 5 </down> ///
		1,0,4,
		4,5,1
	};  

	/// Create Vertex Buffer Object (vbo) ///
	// Generate the buffer indices, and 
	GLuint GenBufferList[2];
	glGenBuffers ( 2, GenBufferList );
	GLuint ArrayBufferData  = GenBufferList [ 0 ];
	GLuint ElementArrayData = GenBufferList [ 1 ];

	// Bind vertex array list
	glBindBuffer ( GL_ARRAY_BUFFER, ArrayBufferData );
	glBufferData ( GL_ARRAY_BUFFER, 24 * sizeof(GLfloat), vertices, GL_STATIC_DRAW );

	// Bind element array list
	glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, ElementArrayData );
	glBufferData ( GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLuint), indices, GL_STATIC_DRAW );

	/// vbo finished ///

	/// Upload attributes of vertex ///
	// Use a cluster for keeping the attributes of vertex
	SGHANDLER cluster;
	glGenVertexArrays ( 1, &cluster );
	glBindVertexArray ( cluster );
	glEnableVertexAttribArray ( 0 );
	glBindBuffer ( GL_ARRAY_BUFFER, ArrayBufferData );
	glVertexAttribPointer ( 0, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat *)NULL );
	glBindBuffer ( GL_ELEMENT_ARRAY_BUFFER, ElementArrayData );  

	cout << "agent object created" << endl;
	return cluster;
};


/* 渲染数据 */
void Framework_v1_0::RenderingFace ( GLenum cullFace, FLUIDSPARAM *fluid )
{
	GLfloat angle  = fluid->ray.nAngle;
	GLuint program = fluid->shader.hProgram;
	GLuint cluster = fluid->ray.hCluster;
	GLuint width   = fluid->ray.uCanvasWidth;
	GLuint height  = fluid->ray.uCanvasHeight;	
	
	// Clear background color and depth buffer
    glClearColor ( 0.0f, 0.f, 0.0f, 0.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    
	//  Set projection and lookat matrix
    mat4 projection = perspective ( 60.0f, (GLfloat)width/height, 0.1f, 400.f );
    mat4 view = lookAt (
		vec3(0.0f, 0.0f, 2.0f),
		vec3(0.0f, 0.0f, 0.0f), 
		vec3(0.0f, 1.0f, 0.0f));

	// Set model view matrix
    mat4 model = mat4(1.0f);
	model = model * rotate ( (float)angle, vec3(0.0f, 1.0f, 0.0f) );
	model = model * translate ( vec3(-0.5f, -0.5f, -0.5f) ); 
    
	// Finally, we focus on setting the Model View Projection Matrix (MVP matrix)
	// Notice that the matrix multiplication order: reverse order of transform
    mat4 mvp = projection * view * model;

	// Returns an integer that represents the location of a specific uniform variable within a shader program
    GLuint mvpIdx = glGetUniformLocation ( program, "mvp" );
    
	if ( mvpIdx >= 0 )
    {
    	glUniformMatrix4fv ( mvpIdx, 1, GL_FALSE, &mvp[0][0] );
    }
    else
    {
    	cout << "can't get the MVP" << endl;
    }
	    
	// Draw agent box
	glEnable ( GL_CULL_FACE );
	glCullFace ( cullFace );
	glBindVertexArray ( cluster );
	glDrawElements ( GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint *)NULL );
	glDisable ( GL_CULL_FACE );
}


/* 向Shader中传递参数 */
void Framework_v1_0::SetVolumeInfoUinforms ( FLUIDSPARAM *fluid )
{
	GLuint program    = fluid->shader.hProgram;
	GLuint Tex1DTrans = fluid->textures.hTexture1D;
	GLuint Tex2DBF    = fluid->textures.hTexture2D;
	GLuint Tex3DVol   = fluid->textures.hTexture3D;
	GLfloat width     = fluid->ray.uCanvasWidth;
	GLfloat height    = fluid->ray.uCanvasHeight;
	GLfloat stepsize  = fluid->ray.fStepsize;

	// Set the uniform of screen size
    GLint screenSizeLoc = glGetUniformLocation ( program, "screensize" );
    if ( screenSizeLoc >= 0 )
    {
		// Incoming two value, width and height
		glUniform2f ( screenSizeLoc, width, height );
    }
    else
    {
		cout << "ScreenSize is not bind to the uniform" << endl;
    }

	// Set the step length
    GLint stepSizeLoc = glGetUniformLocation ( program, "stride" );
	if ( stepSizeLoc >= 0 )
    {
		// Incoming one value, the step size
		glUniform1f ( stepSizeLoc, stepsize );
    }
    else
    {
		cout << "StepSize is not bind to the uniform" << endl;
    }
    
	// Set the transfer function
	GLint transferFuncLoc = glGetUniformLocation ( program, "transfer" );
    if ( transferFuncLoc >= 0 )
	{
		glActiveTexture ( GL_TEXTURE0 );
		glBindTexture ( GL_TEXTURE_1D, Tex1DTrans );
		glUniform1i ( transferFuncLoc, 0 );
    }
    else
    {
		cout << "TransferFunc is not bind to the uniform" << endl;
    }

	// Set the back face as exit point for ray casting
	GLint backFaceLoc = glGetUniformLocation ( program, "stopface" );
	if ( backFaceLoc >= 0 )
    {
		glActiveTexture ( GL_TEXTURE1 );
		glBindTexture(GL_TEXTURE_2D, Tex2DBF);
		glUniform1i(backFaceLoc, 1);
    }
    else
    {
		cout << "ExitPoints is not bind to the uniform" << endl;
    }

	// Set the uniform to hold the data of volumetric data
	GLint volumeLoc = glGetUniformLocation(program, "volumetric");
	if (volumeLoc >= 0)
    {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_3D, Tex3DVol);
		glPixelStorei(GL_UNPACK_ALIGNMENT,1);
		glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, 
			fluid->volume.uWidth, fluid->volume.uHeight, fluid->volume.uDepth,
			0, GL_LUMINANCE, GL_UNSIGNED_BYTE, fluid->volume.ptrData);
		glUniform1i(volumeLoc, 2);
    }
    else
    {
		cout << "VolumeTex is not bind to the uniform" << endl;
    }    
};


/**
***************************** 以上成员函数为进行GLSL渲染所必须初始步骤 ***************************************
***********************************************************************************************************
**************** 以下成员函数包含模型的初始化，及运行。关于流体模拟的计算过程则被封装在其他类中 *****************
*/
#pragma endregion

#include <stdarg.h>
#include <memory>
#include <string>

using std::string;

static SGMAINACTIVITY   *m_activity;
static FLUIDSPARAM       m_fluid;
static FluidSimProc     *m_simproc;

/* 基本框架所默认的构造函数，需要传入SGGUI的地址，以及创建的窗口的长和宽 */
Framework_v1_0::Framework_v1_0( SGMAINACTIVITY **activity, SGUINT width, SGUINT height  )
{
	/* 设置参数 */
	SetDefaultParam();

	/* prepare the fluid simulation stage */
	m_activity = new SGMAINACTIVITY( width, height, false );
	m_simproc  = new FluidSimProc( &m_fluid );
	*activity = m_activity;
	cout << "initial stage finished" << endl;
};


/* 设置模拟程序所需要的一切参数 */
SGVOID Framework_v1_0::SetDefaultParam( SGVOID )
{
	m_fluid.run = true;

	m_fluid.ray.fStepsize     = STEPSIZE;
	m_fluid.ray.nAngle        = 0;
	m_fluid.ray.uCanvasWidth  = CANVAS_X;
	m_fluid.ray.uCanvasHeight = CANVAS_X;

	m_fluid.volume.uWidth     = VOLUME_X;
	m_fluid.volume.uHeight    = VOLUME_X;
	m_fluid.volume.uDepth     = VOLUME_X;

	m_fluid.shader.szCanvasVert = ".\\shader\\backface.vert";
	m_fluid.shader.szCanvasFrag = ".\\shader\\backface.frag";
	m_fluid.shader.szVolumVert  = ".\\shader\\raycasting.vert";
	m_fluid.shader.szVolumFrag  = ".\\shader\\raycasting.frag";
};


/* 对字符串数据的格式化 */
string Framework_v1_0::string_fmt( const std::string fmt_str, ... )
{
	/* reserve 2 times as much as the length of the fmt_str */
    int final_n, n = fmt_str.size() * 2; 
    std::string str;
    std::unique_ptr<char[]> formatted;
    va_list ap;
    while ( true )
	{
		/* wrap the plain char array into the unique_ptr */
        formatted.reset ( new char[n] ); 
        strcpy ( &formatted[0], fmt_str.c_str() );
        va_start ( ap, fmt_str );
        final_n = vsnprintf ( &formatted[0], n, fmt_str.c_str(), ap );
        va_end ( ap );
        if ( final_n < 0 or final_n >= n )
            n += abs( final_n - n + 1 );
        else
            break;
    }
    return std::string ( formatted.get() );
};


/* 创建子线程，用于流体模拟 */
DWORD WINAPI Framework_v1_0::FluidSimulationProc( LPVOID lpParam )
{
	/* 只要m_fluid.run为真，则一直保持流体模拟程序的运行 */
	while ( m_fluid.run ) m_simproc->FluidSimSolver( &m_fluid );

	/* 程序结束，返回 */
	return 0;
};


/* SGGUI运行后，所调度的第一个函数 */
void Framework_v1_0::onCreate()
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
            NULL,                          // default security attributes
            0,                             // use default stack size  
            FluidSimulationProc,           // thread function name
            NULL,                          // argument to thread function 
            0,                             // use default creation flags 
			&m_fluid.thread.dwThreadId);   // returns the thread identifier

	if ( m_fluid.thread.hThread == NULL )
	 {
		 cout << "create sub-thread failed" << endl;
		 exit (1);
	 }

	/* initialize the shader program and textures */
	CreateShaderProg ( &m_fluid );
	m_fluid.textures.hTexture1D   = Create1DTransFunc ( DefaultTransFunc () );
	m_fluid.textures.hTexture2D   = Create2DCanvas ( &m_fluid );
	m_fluid.textures.hTexture3D   = Create3DVolumetric ();
	m_fluid.ray.hCluster          = CreateVerticesBufferObj ();
	m_fluid.textures.hFramebuffer = Create2DFrameBuffer ( &m_fluid );

	/* 打印操作信息 */
	m_simproc->PrintMSG();
};

void Framework_v1_0::CountFPS()
{
	/* finally, print the message on the tile bar */
	SetWindowText( m_activity->GetHWND(), string_fmt( *m_simproc->GetTitleBar(), m_fluid.fps.uFPS ).c_str() );
}

void Framework_v1_0::onDisplay()
{
	/* do something before rendering */
	glEnable ( GL_DEPTH_TEST );
	
	/* bind the vertex buffer object to shader with attribute "vertices" */
	glBindAttribLocation ( m_fluid.shader.hProgram, 0, "vertices" );

    /* do Render Now! */
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, m_fluid.textures.hFramebuffer );
	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hBFVert, m_fluid.shader.hBFFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	RenderingFace ( GL_FRONT, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	/* do not bind the framebuffer now */
    glBindFramebuffer ( GL_FRAMEBUFFER, 0 );

	glViewport ( 0, 0, m_fluid.ray.uCanvasWidth, m_fluid.ray.uCanvasHeight );
	m_fluid.shader.ptrShader->LinkShaders 
		( m_fluid.shader.hProgram, 2, m_fluid.shader.hRCVert, m_fluid.shader.hRCFrag );
	m_fluid.shader.ptrShader->ActiveProgram ( m_fluid.shader.hProgram );
	SetVolumeInfoUinforms ( &m_fluid );
	RenderingFace ( GL_BACK, &m_fluid );
	m_fluid.shader.ptrShader->DeactiveProgram ( m_fluid.shader.hProgram );

	CountFPS ();
};

void Framework_v1_0::onDestroy()
{
	/* 启动退出程序的命令后，先关闭子线程 */
	m_fluid.run = false;
	WaitForSingleObject( m_fluid.thread.hThread, INFINITE );
	CloseHandle( m_fluid.thread.hThread );

	/* 释放所有用于流体计算的资源 */
	m_simproc->FreeResource();

	/* 释放其他资源 */
	SAFE_FREE_PTR( m_simproc );
	SAFE_FREE_PTR( m_fluid.shader.ptrShader );

	/* 打印信息，并退出 */
	cout << "memory freed, program exits..." << endl;
	exit(1);
};

void Framework_v1_0::onKeyboard( SGKEYS keys, SGKEYSTATUS status )
{
	if ( status == SGKEYSTATUS::SG_KEY_DOWN )
	{
		switch (keys)
		{
		case SG_KEY_Q:
		case SG_KEY_ESCAPE:
			onDestroy();
			break;
	
		case SG_KEY_C:
			m_simproc->ZeroBuffers();
			break;

		default:
			break;
		}
	}
};


void Framework_v1_0::onMouse( SGMOUSE mouse, unsigned x, unsigned y, int degree )
{
	if ( mouse eqt SGMOUSE::SG_MOUSE_WHEEL_FORWARD or mouse eqt SGMOUSE::SG_MOUSE_WHEEL_BACKWARD )
	{
		m_fluid.ray.nAngle = (m_fluid.ray.nAngle + degree) % 360;
	}
};