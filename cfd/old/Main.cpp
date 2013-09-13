#include "Headers.h"
#include "resource.h"

#define DISPLAY_WIDTH  800
#define DISPLAY_HEIGHT 600

using namespace sge;

Visualization *visual;
MainActivity  *myact;

extern DWORD simulation (LPVOID lpdwThreadParam);

/* Encapsulate init function */
void OnCreate() { visual->Init(DISPLAY_WIDTH, DISPLAY_HEIGHT); }


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
	myact  = new MainActivity(DISPLAY_WIDTH, DISPLAY_HEIGHT);
	visual = new Visualization(myact);

	// Set app icons
	myact->SetApplicationIcons(IDI_ICON1, IDI_ICON1);

	// Register functions
	myact->RegisterCreateFunc(OnCreate);
	myact->RegisterDisplayFunc(visual->Display);
	myact->RegisterKeyboardFunc(visual->Keyboard);
	myact->RegisterMouseFunc(visual->Mouse);

	// Setup MFC window
	myact->SetupRoutine();

	// Mainloop
	myact->MainLoop();

	// Release all resource
	SAFE_DELT_PTR(visual);
	SAFE_DELT_PTR(myact);

	return 0;
}