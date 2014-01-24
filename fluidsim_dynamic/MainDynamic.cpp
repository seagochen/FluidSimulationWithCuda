/**
* <Author>      Orlando Chen
* <First>       Oct 16, 2013
* <Last>		Jan 25, 2014
* <File>        MainDynamic.cpp
*/

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <GL\glew32c.h>
#include <SGE\SGUtils.h>
#include "resource.h"
#include "Parameter.h"
#include "MainFrameworkDynamic.h"

using sge::SGMAINACTIVITY;
using sge::FrameworkDynamic;

int main()
{
	SGMAINACTIVITY  *activity;
	FrameworkDynamic framework;

	/* create instance of activity */
	activity = new SGMAINACTIVITY( WINDOWS_X, WINDOWS_X, false );

	/* set icon */
	activity->SetAppClientInfo( IDI_ICON1, IDI_ICON1 );

	/* set callback functions */
	activity->RegisterCreateFunc( framework.onCreate );
	activity->RegisterDestroyFunc( framework.onDestroy );
	activity->RegisterDisplayFunc( framework.onDisplay );
	activity->RegisterReshapeFunc( framework.onReshape );
	activity->RegisterKeyboardFunc( framework.onKeyboard );
	activity->RegisterMouseFunc( framework.onMouse );
	activity->RegisterIdleFunc( framework.onIdle );
};