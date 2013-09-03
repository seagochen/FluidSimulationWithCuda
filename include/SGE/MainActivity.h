/**
* ------------------------------------the MIT License--------------------------------------------------
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
* <Date>        8/25/2013
* <File>        MainActivity.h
* <Specification>
*	與程序初始化有關的類，包含對MFC的封裝
*	- 利用MFC創建窗口；
*	- 利用MFC監聽操作事件；
*/

#ifndef _SEAGOSOFT_ENGINE_MAINACTIVITY_H_
#define _SEAGOSOFT_ENGINE_MAINACTIVITY_H_

#include <Windows.h>

#ifdef _In_Dll_File
#include "GameEvents.h"
#endif

namespace sge
{
	class MainActivity
	{
	public:
		// Default Constructor
		_DLL MainActivity(bool isFull = false);
		// Constructor with specified window's size
		_DLL MainActivity(unsigned width, unsigned height, bool isFull = false);

	public:
		// Register create (initialize) function
		_DLL static void RegisterCreateFunc(
			void (*hCreateFunc)(void));
		// Register reshape function
		_DLL static void RegisterReshapeFunc(
			void (*hReshapeFunc)(unsigned width, unsigned height));
		// Register keyboard function
		_DLL static void RegisterKeyboardFunc(
			void (*hKeyboardFunc)(SG_KEYS keyboard, SG_KEY_STATUS keystate));
		// Register mouse function
		_DLL static void RegisterMouseFunc(
			void (*hMouseFunc)(SG_MOUSE mouse, unsigned xpos, unsigned ypos));
		// Register display function
		_DLL static void RegisterDisplayFunc(
			void (*hDisplayFunc)(void));

	public:
		// When functions registered call this function to create app form
		_DLL static int SetupRoutine(void);
		// Display the app form and loop the message
		_DLL static int MainLoop(void);

	private:
		_DLL static LRESULT CALLBACK WindowProc(HWND hwnd,UINT message,WPARAM wParam,LPARAM lParam);

	public:
		_DLL HDC       GetHDC();
		_DLL HINSTANCE GetHInstance();
		_DLL HGLRC     GetHRC();
		_DLL HWND      GetHWND();

	public:
		_DLL void  SetApplicationTitle(LPCWSTR szTitle);
		_DLL void  SetApplicationIcons(WORD wAppIcon, WORD wSmallIcon);
		_DLL void  ConvertMFCPosition(unsigned *xpos_in_out, unsigned *ypos_in_out);
	};
};

#endif