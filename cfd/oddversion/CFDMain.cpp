#include "Visual.h"
#include "resource.h"

#define DISPLAY_WIDTH  800
#define DISPLAY_HEIGHT 600

using namespace sge;

Visual       *visual;
MainActivity *activity;

extern DWORD simulation (LPVOID lpdwThreadParam);


/* Create a thread for computational fluid dynamic  */
void CreateSubThread()
{
	DWORD dwThreadId;
	
	// Create thread in MFC
	if ( CreateThread(NULL,                   //Choose default security
		0,                                    //Default stack size
		(LPTHREAD_START_ROUTINE)&simulation,  //Routine to execute
		0,                                    //Thread parameter
		0,                                    //Immediately run the thread
		&dwThreadId ) == NULL)                //Thread Id	
		{
			ErrorMSG("Error in line %d: Couldn't creat sub-thread. Aborting.\n");
		}
}

int main()
{
	// Create thread
	CreateSubThread();
	
	// Set window size
	activity  = new MainActivity(DISPLAY_WIDTH, DISPLAY_HEIGHT);
	visual    = new Visual(DISPLAY_WIDTH, DISPLAY_HEIGHT, activity);

	// Set app icons
	activity->SetApplicationIcons(IDI_ICON1, IDI_ICON1);

	// Register functions
	activity->RegisterCreateFunc(visual->OnCreate);
	activity->RegisterDisplayFunc(visual->OnDisplay);
	activity->RegisterKeyboardFunc(visual->OnKeyboard);
	activity->RegisterMouseFunc(visual->OnMouse);
//	activity->RegisterDestoryFunc(visual->OnDestroy);

	// Setup MFC window
	activity->SetupRoutine();

	// Mainloop
	activity->MainLoop();

	// Release all resource
	SAFE_DELT_PTR(visual);
	SAFE_DELT_PTR(activity);

	return 0;
}