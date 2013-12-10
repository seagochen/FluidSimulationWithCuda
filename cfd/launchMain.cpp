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
* <First>       Oct 6, 2013
* <Last>		Nov 18, 2013
* <File>        launchMain.cpp
*/

#ifndef __launch_main_cpp_
#define __launch_main_cpp_

#include "cfdHeader.h"

#define APP_ICONS IDI_ICON1

using namespace sge;

MainActivity  *activity;
Visualization *visual;

/*
-----------------------------------------------------------------------------------------------------------
* @function main
* @author   Orlando Chen
* @date     Nov 18, 2013
* @input    NULL
* @return   NULL
* @bref     Main function      
-----------------------------------------------------------------------------------------------------------
*/
int main(int argc, char ** argv)
{
	// Create a main activity and set the window from size as 512x512
//	activity = new MainActivity(Client_X, Client_X);
	visual   = new Visualization ( Client_X, Client_X, &activity );

	// Set application title
	activity->SetAppClientInfo ( L"Navigator - 1930b", APP_ICONS, APP_ICONS );
	//activity->SetApplicationTitle ( L"Navigator - 1930b" );
	//activity->SetApplicationIcons ( APP_ICONS, APP_ICONS );
			
	// Register callback functions
	activity->RegisterCreateFunc   ( visual->OnCreate   );
	activity->RegisterDestoryFunc  ( visual->OnDestroy  );
	activity->RegisterKeyboardFunc ( visual->OnKeyboard );
	activity->RegisterMouseFunc    ( visual->OnMouse    );
	activity->RegisterReshapeFunc  ( visual->OnResize   );
	activity->RegisterDisplayFunc  ( visual->OnDisplay  );
	activity->RegisterIdleFunc     ( visual->OnIdle     );

	// Initialize window
	activity->SetupRoutine();

	return 0;
};

#endif