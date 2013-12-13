#ifndef __main_cpp_
#define __main_cpp_

#include <GL\glew.h>
#include <GL\freeglut.h>

//#include <SGE\SGUtils.h>

#include <string>
#include <iostream>

#include <SGE\SGUtils.h>

using namespace sge;
using namespace std;

void onCreate ()
{


	goto Success;

Error:
	system ("pause");
	exit (1);

Success:
	;
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