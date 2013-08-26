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

#include "GameEvents.h"

namespace sge
{
	class MainActivity
	{
		struct SGFUNCTIONSHANDLER
		{
			void  (*hCreateFunc)(void);
			void  (*hReshapeFunc)(int width, int height);
			void  (*hKeyDownFunc)(SGKEYS key);
			void  (*hKeyUpFunc)(SGKEYS key);
			void  (*hDisplayFunc)(void);
		};

	public:
		SGFUNCTIONSHANDLER SG_FuncsHolder;

	public:
		// Register create (initialize) function
		DllExport void RegisterCreateFunc(void  (*hCreateFunc)(void));
		// Register reshape function
		DllExport void RegisterReshapeFunc(void  (*hReshapeFunc)(int width, int height));
		// Register keyboard function, when key is be pressed down
		DllExport void RegisterKeyDownFunc(void  (*hKeyFunc)(SGKEYS key));
		// Register keyboard function, when key is be released
		DllExport void RegisterKeyUpFunc(void  (*hKeyFunc)(SGKEYS key));
		// Register display function
		DllExport void RegisterDisplayFunc(void  (*hDisplayFunc)(void));

	private:
		RECT         windowRect;
		bool32       isFullScreenMode;
		HINSTANCE    hInstance;

	public:
		// Default Constructor
		DllExport MainActivity(bool32 isFull = false);
		// Constructor with specified window's size
		DllExport MainActivity(int32 width, int32 height, bool32 isFull = false);
		// Constructor with windows'size and title
		DllExport MainActivity(int32 width, int32 height, const LPCSTR title, bool32 isFull = false);

	private:
		HWND         hwnd;
		HDC          hDC;
		HGLRC        hRC;

	public:
		DllExport inline void setHWND(HWND hWND_in) { hwnd = hWND_in; };
		DllExport inline void setHDC(HDC hDC_in) { hDC = hDC_in; };
		DllExport inline void setHGLRC(HGLRC hRC_in) { hRC = hRC_in; };
		DllExport inline void setHINSTANCE(HINSTANCE hInstance_in) { hInstance = hInstance_in; };

		DllExport inline HWND getHWND(void) { return hwnd; };
		DllExport inline HDC  getHDC(void) { return hDC; };
		DllExport inline HGLRC getHGLRC(void) { return hRC; };
		DllExport inline HINSTANCE getInstance() { return hInstance; };

	public:
		int32 SetupRoutine(const LPCSTR title);
		int32 WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
		int32 MainLoop();

	public:
		inline bool32 isFullScreen() { return isFullScreenMode; };
		inline void setFullScreen(bool32 mode) { isFullScreenMode = mode; };

		inline RECT *getWindowRect() { return &windowRect; };
	};
};

#endif