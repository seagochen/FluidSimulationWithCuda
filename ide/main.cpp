#ifndef __main_cpp_
#define __main_cpp_

#include "glew32.h"
#include <GL\freeglut.h>

//#include <SGE\SGUtils.h>

#include <string>
#include <iostream>

#include <SGE\SGUtils.h>


void onCreate ()
{
	using namespace sge;

	GLenum err = glewInit();
	if ( GLEW_OK != err )
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		FileManager::Write ( "runtime.log", SG_APPEND, 
			"glewInit failed: %s", glewGetErrorString(err));
		exit (1);
	}
	else
	{
		FileManager::Write ( "runtime.log", SG_APPEND, "glewInit successed" );
	}

	std::string str;
	char *msg = 0;
	FileManager::Read ( ".\\Shader\\simple.vert", &str );
	FileManager::Read ( ".\\Shader\\simple.vert", 10000, &msg);

	using namespace std;

	cout<<"----------"<<endl;
	cout<<str<<endl<<endl;
	cout<<"----------"<<endl;
	cout<<msg<<endl;
};


void onDisplay ()
{
};


void onIdle ()
{
};


int main()
{
	using namespace sge;

	MainActivity *activity = new MainActivity(800, 600);

	activity->SetAppClientInfo(L"Hello World!");

	activity->RegisterCreateFunc(onCreate);
	activity->RegisterDisplayFunc(onDisplay);
	activity->RegisterIdleFunc(onIdle);
	
	activity->SetupRoutine();
}

#endif