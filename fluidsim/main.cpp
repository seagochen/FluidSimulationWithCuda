#define INCLUDE_GLEW32_CODE

#include "main.h"

GLuint prog, bfVert, bfFrag, rcVert, rcFrag;
Shader *shader;


void onCreate ()
{
	CreateShaders ( shader, &prog, &bfVert, &bfFrag, &rcVert, &rcFrag );

	// Specify the background clearing color
	glClearColor(0.7f, 0.7f, 0.7f, 0.5f);

	cout << "initialize finished, sge will work soon!" << endl;
};



void onDisplay ()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
		
	gluLookAt(0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

//	shader.ActiveProgram ( prog );

	// painting a quad
	glBegin(GL_QUADS);
	   glVertex3f(-0.5, -0.5, 0.0);
	   glVertex3f(0.5, -0.5, 0.0);
	   glVertex3f(0.5, 0.5, 0.0);
	   glVertex3f(-0.5, 0.5, 0.0);
	glEnd();

//	shader.DeactiveProgram ( prog );
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