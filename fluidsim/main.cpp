/**
*
* Copyright (C) <2013> <Orlando Chen>
* Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
* associated documentation files (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
* and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all copies or substantial
* portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT 
* NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
* <Author>      Orlando Chen
* <First>       Dec 12, 2013
* <Last>		Dec 18, 2013
* <File>        main.cpp
*/

#define __main_cpp_

#include <GL\glew32c.h>
#include <GL\freeglut.h>

#include "funcdef.h"
#include "cfd.h"

fluidsim fluid;


void initParam ()
{
	fluid.nScrHeight = 400;  // default height is 400 px
	fluid.nScrWidth  = 400;  // default width is 400 px
//	fluid.nVolWidth  = 256;  // default volume width is 256
//	fluid.nVolHeight = 256;  // default volume height is 256
//	fluid.nVolDepth  = 225;  // default volume depth is 225
	fluid.nVolWidth  = Grids_X;
	fluid.nVolHeight = Grids_X;
	fluid.nVolDepth  = Grids_X;

	fluid.fStepsize  = 0.001f;
	fluid.nAngle     = 0;

	fluid.ptrShader  = new Shader();

	cout << "parameters initialize finished" << endl;
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
	CreateShaders ( &fluid );

	// Create texture objects
	fluid.hTexture1D = Create1DTransFunc ();
	fluid.hTexture2D = Create2DBackFace ( &fluid );
	//fluid.hTexture3D = Create3DVolumetric ( ".\\res\\head256.raw", &fluid );
	fluid.hTexture3D = Create3DVolumetric ();

	// Create vertex buffer
	fluid.hCluster = InitVerticesBufferObj ();

	// Create frame buffer
	fluid.hFramebuffer = CreateFrameBuffer ( &fluid );

	cout << "initialize finished, sge will work soon!" << endl;
};



void onDisplay ()
{
	glEnable ( GL_DEPTH_TEST );
	
	// Bind index 0 to the shader input variable "VerPos"
	glBindAttribLocation ( fluid.hProgram, 0, "vertices" );

    /// Do Render Now!
	glBindFramebuffer ( GL_DRAW_FRAMEBUFFER, fluid.hFramebuffer ); // Chose which framebuffer to render
	glViewport ( 0, 0, fluid.nScrWidth, fluid.nScrHeight );
	fluid.ptrShader->LinkShaders ( fluid.hProgram, 2, fluid.hBFVert, fluid.hBFFrag );
	fluid.ptrShader->ActiveProgram ( fluid.hProgram );
	RenderingFace ( GL_FRONT, &fluid );   // From front face
	fluid.ptrShader->DeactiveProgram ( fluid.hProgram );


    glBindFramebuffer ( GL_FRAMEBUFFER, 0 ); // To break the binding
    glViewport ( 0, 0, fluid.nScrWidth, fluid.nScrHeight );
	fluid.ptrShader->LinkShaders ( fluid.hProgram, 2, fluid.hRCFrag, fluid.hRCVert );
	fluid.ptrShader->ActiveProgram ( fluid.hProgram );
	SetVolumeInfoUinforms ( &fluid );
	RenderingFace ( GL_BACK, &fluid );
	fluid.ptrShader->DeactiveProgram ( fluid.hProgram );

	fluid.nAngle = (fluid.nAngle + 1) % 360;
};


void onIdle ()
{
	// Solving the density
	DensitySolver ( host_den, host_den0, host_u, host_v, host_w );
	// Solving the velocity
	VelocitySolver ( host_u, host_v, host_w, host_u0, host_v0, host_w0 );
	// Interpolation
	DensityItp ( host_den, dataset );

	fluid.ptrData = dataset;
};


int main()
{
	// initialize parameters
	initParam ();

	// initialize the device array and push them into stack
	if ( AllocateResourcePtrs () != SG_RUNTIME_OK ) 
	{
		cout << "allocate the memory and cache failed" << endl;
		exit (1);
	}
	else
	{
		ZeroData ();
		// Prepare the stage for fluid simulation
		InitSimGrid ( host_den, host_u, host_v, host_w );
		cout << "fluid simulation stage prepared" << endl;
	}

	MainActivity *activity = new MainActivity ( fluid.nScrWidth, fluid.nScrHeight );

	activity->SetAppClientInfo ( L"Excalibur OTL 0.00.00.001" );

	activity->RegisterCreateFunc ( onCreate );
	activity->RegisterIdleFunc ( onIdle );
	activity->RegisterDisplayFunc ( onDisplay );
	
	activity->SetupRoutine ();
}