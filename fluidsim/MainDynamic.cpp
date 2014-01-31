/**
* <Author>      Orlando Chen
* <First>       Oct 16, 2013
* <Last>		Jan 25, 2014
* <File>        MainDynamic.cpp
*/

#include <GL\glew.h>
#include <GL\glew32c.h>
#include <SGE\SGUtils.h>
#include "resource.h"
#include "MainFrameworkDynamic.h"

int main()
{
	using namespace sge;
	SGMAINACTIVITY *activity;
	FrameworkDynamic famework( &activity, WINDOWS_X, WINDOWS_X );

	activity->SetAppClientInfo ( IDI_ICON1, IDI_ICON1 );
	activity->RegisterCreateFunc ( famework.onCreate );
	activity->RegisterDisplayFunc ( famework.onDisplay );
	activity->RegisterMouseFunc ( famework.onMouse );
	activity->RegisterDestroyFunc ( famework.onDestroy );
	activity->RegisterKeyboardFunc ( famework.onKeyboard );
	
	activity->SetupRoutine();
}