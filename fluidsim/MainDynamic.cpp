/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Oct 16, 2013
* <Last Time>     Feb 02, 2014
* <File Name>     MainDynamic.cpp
*/

#ifndef __main_dynamic_cpp_
#define __main_dynamic_cpp_

#pragma once

#include <GL\glew.h>
#include <GL\glew32c.h>
#include <SGE\SGUtils.h>
#include "resource.h"
#include "MainFrameworkDynamic.h"

using namespace sge;

SGMAINACTIVITY *activity;

int main()
{
	/* 使用基本框架并对SGGUI进行初始化 */
	FrameworkDynamic famework( &activity, WINDOWS_X, WINDOWS_X );

	/* 对SGGUI的设置 */
	activity->SetAppClientInfo     ( IDI_ICON1, IDI_ICON1 );
	activity->RegisterCreateFunc   ( famework.onCreate );
	activity->RegisterDisplayFunc  ( famework.onDisplay );
	activity->RegisterMouseFunc    ( famework.onMouse );
	activity->RegisterDestroyFunc  ( famework.onDestroy );
	activity->RegisterKeyboardFunc ( famework.onKeyboard );
	
	/* 启动SGGUI */
	activity->SetupRoutine();
};

#endif