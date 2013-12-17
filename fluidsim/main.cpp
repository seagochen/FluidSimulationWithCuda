#include <GL\glew32c.h>
#include <GL\freeglut.h>

#include "main.h"

fluidsim fluid;


void initParam ()
{
	fluid.nScrHeight = 400;  // default height is 400 px
	fluid.nScrWidth  = 400;  // default width is 400 px
	fluid.nVolWidth  = 256;  // default volume width is 256
	fluid.nVolHeight = 256;  // default volume height is 256
	fluid.nVolDepth  = 225;  // default volume depth is 225

	fluid.fStepsize  = 0.001f;
	fluid.nAngle     = 0;

	fluid.ptrShader  = new Shader();
}


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

	// Create shader and program objects
	CreateShaders ( fluid.ptrShader, &fluid.hProgram, 
		&fluid.hBFVert, &fluid.hBFFrag, &fluid.hRCVert, &fluid.hRCFrag );

	// Create texture objects
	fluid.hTexture1D = Create1DTransFunc ();
	fluid.hTexture2D = Create2DBackFace ( fluid.nScrWidth, fluid.nScrHeight );
	fluid.hTexture3D = Create3DVolumetric ( ".\\res\\head256.raw", fluid.nVolWidth, fluid.nVolHeight, fluid.nVolDepth  );

	// Create vertex buffer
	fluid.hCluster = InitVerticesBufferObj ();

	// Create frame buffer
	fluid.hFramebuffer = CreateFrameBuffer ( fluid.hTexture2D, fluid.nScrWidth, fluid.nScrHeight );

	cout << "initialize finished, sge will work soon!" << endl;
};



void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	// Bind index 0 to the shader input variable "VerPos"
	glBindAttribLocation ( fluid.hProgram, 0, "VerPos" );
	// Bind index 1 to the shader input variable "VerClr"
	glBindAttribLocation ( fluid.hProgram, 1, "VerClr" );

    /// Do Render Now!
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, fluid.hFramebuffer ); // Chose which framebuffer to render
	glViewport ( 0, 0, fluid.nScrWidth, fluid.nScrHeight );
	fluid.ptrShader->LinkShaders ( fluid.hProgram, 2, fluid.hBFVert, fluid.hBFFrag );
	fluid.ptrShader->ActiveProgram ( fluid.hProgram );
	RenderingFace ( GL_FRONT, fluid.nAngle, fluid.hProgram, fluid.hCluster, fluid.nScrWidth, fluid.nScrHeight );   // From front face
	fluid.ptrShader->DeactiveProgram ( fluid.hProgram );


    glBindFramebuffer ( GL_FRAMEBUFFER, 0 ); // To break the binding
    glViewport ( 0, 0, fluid.nScrWidth, fluid.nScrHeight );
	fluid.ptrShader->LinkShaders ( fluid.hProgram, 2, fluid.hRCFrag, fluid.hRCVert );
	fluid.ptrShader->ActiveProgram ( fluid.hProgram );
	SetVolumeInfoUinforms ( fluid.hProgram, fluid.hTexture1D, fluid.hTexture2D, fluid.hTexture3D, fluid.nScrWidth, fluid.nScrHeight, fluid.fStepsize );
	RenderingFace ( GL_BACK, fluid.nAngle, fluid.hProgram, fluid.hCluster, fluid.nScrWidth, fluid.nScrHeight );
	fluid.ptrShader->DeactiveProgram ( fluid.hProgram );

	fluid.nAngle = (fluid.nAngle + 1) % 360;
};


int main()
{
	// initialize parameters
	initParam ();

	MainActivity *activity = new MainActivity ( fluid.nScrWidth, fluid.nScrHeight );

	activity->SetAppClientInfo ( L"Excalibur OTL 0.00.00.001" );

	activity->RegisterCreateFunc ( onCreate );
	activity->RegisterDisplayFunc ( onDisplay );
	
	activity->SetupRoutine ();
}