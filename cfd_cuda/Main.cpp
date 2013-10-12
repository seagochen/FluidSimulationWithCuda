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
* <Last>		Oct 10, 2013
* <File>        Main.cpp
*/


#define _IN_MAIN_FILE
#include <GL\glew.h>
#include <GL\glut.h>
#include <SGE\SGUtils.h>
#include "Macro_Definitions.h"
#include "resource.h"

////////////////////////////////////////////////////////////////////////
///

using namespace sge;

MainActivity  *activity;
Visualization *visual;

///
////////////////////////////////////////////////////////////////////////
///

int main( int argc, char ** argv )
{
	// Create a main activity and set the window from size as 512x512
	activity = new MainActivity(WINDOWSX, WINDOWSY);
	visual   = new Visualization(WINDOWSX, WINDOWSX, activity);

	// Register callback function to visualization
	visual->RegisterDisplay ( display_func );
	visual->RegisterIdle    ( idle_func );
	visual->RegisterKeyboard( key_func );
	visual->RegisterMouse   ( mouse_func );
	visual->RegisterResize  ( reshape_func );
	visual->RegisterDestroy ( dest_func );

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	// Set application title
	activity->SetApplicationTitle( L"CFD - Navigator No. I" );
	activity->SetApplicationIcons(APP_ICONS, APP_ICONS);
			
	// Register callback functions
	activity->RegisterCreateFunc   ( visual->OnCreate );
	activity->RegisterDestoryFunc  ( visual->OnDestroy );
	activity->RegisterKeyboardFunc ( visual->OnKeyboard );
	activity->RegisterMouseFunc    ( visual->OnMouse );
	activity->RegisterReshapeFunc  ( visual->OnResize );
	activity->RegisterDisplayFunc  ( visual->OnDisplay );
	activity->RegisterIdleFunc     ( visual->OnIdle );

	// Initialize window
	activity->SetupRoutine();

	// Display and run demo
	activity->MainLoop();

	exit ( 0 );
};

///
////////////////////////////////////////////////////////////////////////