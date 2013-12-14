#define __main_cpp_

#include "glew32.h"
#include <GL\freeglut.h>

#include <SGE\SGUtils.h>

#include <string>
#include <iostream>


using namespace sge;
using namespace std;

GLuint vert, frag;
GLuint prog;

Shader shader;


void onCreate ()
{
	// Create shader objects from source
	shader.CreateShaderObj ( ".\\Shader\\simple.vert", SG_VERTEX,   &vert );
	shader.CreateShaderObj ( ".\\Shader\\simple.frag", SG_FRAGMENT, &frag );

	// Check error
	if ( vert == NULL || frag == NULL )
	{
		cout << "create shaders object failed" << endl;
		exit (1);
	}
	
	// Create shader program object
	shader.CreateProgmObj ( &prog );

	if ( prog == NULL )
	{
		cout << "create program object failed" << endl;
		exit (1);
	}
	
	// Attach and link the program
	if ( shader.LinkShaders ( prog, 2, vert, frag ) != SG_RUNTIME_OK )
	{
		cout << "link shader program failed" << endl;
		exit (1);
	}
	
	// Black Background
	glClearColor(0.7f, 0.7f, 0.7f, 0.5f);
};


void onDisplay ()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	// Clear Screen And Depth Buffer
	glLoadIdentity();									// Reset The Current Modelview Matrix
		
	gluLookAt(0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	// bind the GLSL program	
	// this means that from here the GLSL program attends all OpenGL operations
	shader.ActiveProgram ( prog );

	// painting a quad
	glBegin(GL_QUADS);
	   glVertex3f(-0.5, -0.5, 0.0);
	   glVertex3f(0.5, -0.5, 0.0);
	   glVertex3f(0.5, 0.5, 0.0);
	   glVertex3f(-0.5, 0.5, 0.0);
	glEnd();

	shader.DeactiveProgram ( prog );
};


void onIdle ()
{
};


void onResize ( unsigned width, unsigned height )
{
	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);     // Select The Projection Matrix
	glLoadIdentity();                // Reset The Projection Matrix
	// Calculate The Aspect Ratio And Set The Clipping Volume
	if (height == 0) height = 1;
	gluPerspective(45, (float)width/(float)height, 0.1, 100.0);
	glMatrixMode(GL_MODELVIEW);      // Select The Modelview Matrix
	glLoadIdentity();                // Reset The Modelview Matrix

};


int main()
{
	using namespace sge;

	MainActivity *activity = new MainActivity(800, 600);

	activity->SetAppClientInfo(L"Hello World!");

	activity->RegisterCreateFunc(onCreate);
	activity->RegisterDisplayFunc(onDisplay);
	activity->RegisterIdleFunc(onIdle);
	activity->RegisterReshapeFunc (onResize);
	
	activity->SetupRoutine();
}