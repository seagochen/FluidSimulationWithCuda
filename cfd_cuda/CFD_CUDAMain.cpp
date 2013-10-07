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
* <Last>		Oct 7, 2013
* <File>        CFD_CUDAMain.cpp
*/
#define _IN_MAIN_FILE
#include <GL\glew.h>
#include <GL\glut.h>
#include <SGE\SGUtils.h>
#include "CFD_Params.h"
#include "CFD_FuncPrototypes.h"
#include "CFD_Visualization.h"
#include "resource.h"

////////////////////////////////////////////////////////////////////////
///

using namespace sge;

MainActivity  *activity;
Visualization *visual;

///
////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Initialize functions
  ----------------------------------------------------------------------
*/


void CFD_Init_Params()
{
	GridSize = 64;
	dt = 0.1f;
	diff = 0.0f;
	visc = 0.0f;
	force = 5.0f;
	source = 100.0f;
	win_x = 512;
	win_y = 512;
};

void CFD_Init_Routine(void)
{
	// Create a main activity and set the window from size as 512x512
	activity = new MainActivity(win_x, win_y);
	visual   = new Visualization(win_x, win_y, activity);

	// Register callback function to visualization
	visual->RegisterDisplay ( display_func );
	visual->RegisterIdle    ( idle_func );
	visual->RegisterKeyboard( key_func );
	visual->RegisterMouse   ( mouse_func );
	visual->RegisterResize  ( reshape_func );
	visual->RegisterDestroy ( dest_func );
};


/*
  ----------------------------------------------------------------------
   Main function
  ----------------------------------------------------------------------
*/

int main( int argc, char ** argv )
{
	// Initialize the CFD parameters & routine
	CFD_Init_Params();
	CFD_Init_Routine();

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	// Set application title
	activity->SetApplicationTitle( L"CFD Demo     Version 1.00.01a" );
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