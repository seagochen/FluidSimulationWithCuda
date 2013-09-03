#include "Headers.h"

#define DISPLAY_WIDTH  800
#define DISPLAY_HEIGHT 600

using namespace sge;

Visualization *visual;
MainActivity  *myact;

extern DWORD simulation (LPVOID lpdwThreadParam );

void OnCreate() { visual->Init(DISPLAY_WIDTH, DISPLAY_HEIGHT); }

/* 创建两个不同的子线程，一个负责计算并生成volume data，另一个则负责计算FPS */
void subThreads()
{
	DWORD dwThreadId;
	
	// 创建计算流体的线程
	if ( CreateThread(NULL, //Choose default security
		0, //Default stack size
		(LPTHREAD_START_ROUTINE)&simulation, //Routine to execute
		0, //Thread parameter
		0, //Immediately run the thread
		&dwThreadId ) == NULL) //Thread Id	
		{
			pterror("Error in line %d: Couldn't creat sub-thread. Aborting.\n");
		}
}


int main()
{
	// Creates sub threads
	subThreads();
	
	// Set window size
	myact  = new MainActivity(DISPLAY_WIDTH, DISPLAY_HEIGHT);
	visual = new Visualization(myact);

	// Register functions
	myact->RegisterCreateFunc(OnCreate);
	myact->RegisterDisplayFunc(visual->Display);
	myact->RegisterKeyboardFunc(visual->Keyboard);
	myact->RegisterMouseFunc(visual->Mouse);

	// Setup MFC window
	myact->SetupRoutine();

	// Mainloop
	myact->MainLoop();

	SAFE_DELT_PTR(myact);
	return 0;
}