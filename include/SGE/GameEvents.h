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
* <File>        GameEvents.h
* <Specification>
*	封裝了MFC的消息機制
*/

#ifndef _SEAGOSOFT_ENGINE_GAMEEVENT_H_
#define _SEAGOSOFT_ENGINE_GAMEEVENT_H_

#include "Auxiliaries.h"
#include <Windows.h>
#include <GL\glut.h>

namespace sge {

	enum SG_KEY_STATUS
	{
		SG_KEY_DOWN = WM_KEYDOWN,  // key down
		SG_KEY_UP   = WM_KEYUP,    // key up
		SG_KEY_NONE = 0x00         // Unknow
	};

	enum SG_KEYS
	{
		// Key groups
		SG_KEYS_CANCEL  = VK_CANCEL,   // Ctrl+Break or Ctrl+C

		// 不常用按键
		SG_KEY_CLEAR   = VK_CLEAR,   // Clear
		SG_KEY_SELECT  = VK_SELECT,  // Select
		SG_KEY_EXECUTE = VK_EXECUTE, // Execute
		SG_KEY_HELP    = VK_HELP,    // Help 
		
		// 常用功能按键
		SG_KEY_ESCAPE    = VK_ESCAPE,  // Esc
		SG_KEY_TAB       = VK_TAB,     // Tab
		SG_KEY_ENTER     = VK_RETURN,  // Enter
		SG_KEY_SHIFT     = VK_SHIFT,   // Shift
		SG_KEY_CTRL      = VK_CONTROL, // Ctrl
		SG_KEY_ALT       = VK_MENU,    // Alt
		SG_KEY_SPACE     = VK_SPACE,   // SpaceBar
		SG_KEY_BACKSPACE =  VK_BACK,   // Backspace
		SG_KEY_CAPLOCK   =  VK_CAPITAL,// Capslock
		
		// 一般功能按键
		SG_KEY_INSERT     = VK_INSERT,   // Insert
		SG_KEY_HOME       = VK_HOME,     // Home
		SG_KEY_PAGEDOWN   = VK_NEXT,     // Page Down
		SG_KEY_PAGEUP     = VK_PRIOR,    // Page Up
		SG_KEY_DEL        = VK_DELETE,   // Del
		SG_KEY_END        = VK_END,      // End
		SG_KEY_PAUSE      = VK_PAUSE,    // Pause
		SG_KEY_SNAPSHOT   = VK_SNAPSHOT, // PrintScreen
		SG_KEY_SCROLL     = VK_SCROLL,   // ScrollLock
		
		// 方向键
		SG_KEY_ARRLEFT    = VK_LEFT,  // Left Arrow
		SG_KEY_ARRUP      = VK_UP,    // Up Arrow
		SG_KEY_ARRRIGHT   = VK_RIGHT, // Right Arrow
		SG_KEY_ARRDOWN    = VK_DOWN,  // Down Arrow
		
		//定义数据字符0~9 
		SG_KEY_0 = '0',   // num 0
		SG_KEY_1 = '1',   // num 1
		SG_KEY_2 = '2',   // num 2
		SG_KEY_3 = '3',   // num 3
		SG_KEY_4 = '4',   // num 4
		SG_KEY_5 = '5',   // num 5
		SG_KEY_6 = '6',   // num 6
		SG_KEY_7 = '7',   // num 7
		SG_KEY_8 = '8',   // num 8
		SG_KEY_9 = '9',   // num 9
		
		//定义数据字符A~Z
		SG_KEY_A = 'A',   // A
		SG_KEY_B = 'B',   // B
		SG_KEY_C = 'C',   // C
		SG_KEY_D = 'D',   // D
		SG_KEY_E = 'E',   // E
		SG_KEY_F = 'F',   // F
		SG_KEY_G = 'G',   // G
		SG_KEY_H = 'H',   // H
		SG_KEY_I = 'I',   // I
		SG_KEY_J = 'J',   // J
		SG_KEY_K = 'K',   // K
		SG_KEY_L = 'L',   // L
		SG_KEY_M = 'M',   // M
		SG_KEY_N = 'N',   // N
		SG_KEY_O = 'O',   // O
		SG_KEY_P = 'P',   // P
		SG_KEY_Q = 'Q',   // Q
		SG_KEY_R = 'R',   // R
		SG_KEY_S = 'S',   // S
		SG_KEY_T = 'T',   // T
		SG_KEY_U = 'U',   // U
		SG_KEY_V = 'V',   // V
		SG_KEY_W = 'W',   // W
		SG_KEY_X = 'X',   // X
		SG_KEY_Y = 'Y',   // Y
		SG_KEY_Z = 'Z',   // Z
		
		// F1~F12
		SG_KEY_F1 = VK_F1,  // F1
		SG_KEY_F2 = VK_F2,  // F2
		SG_KEY_F3 = VK_F3,  // F3
		SG_KEY_F4 = VK_F4,  // F4
		SG_KEY_F5 = VK_F5,  // F5
		SG_KEY_F6 = VK_F6,  // F6
		SG_KEY_F7 = VK_F7,  // F7
		SG_KEY_F8 = VK_F8,  // F8
		SG_KEY_F9 = VK_F9,  // F9
		SG_KEY_F10 = VK_F10, // F10
		SG_KEY_F11 = VK_F11, // F11
		SG_KEY_F12 = VK_F12, // F12
	};

	enum SG_MOUSE
	{
		SG_MOUSE_MOVE  = WM_MOUSEMOVE, // mouse is moving
		SG_MOUSE_WHEEL = WM_MOUSEWHEEL, // mouse wheel

		SG_MOUSE_L_BUTTON_DOWN = WM_LBUTTONDOWN, // left button down
		SG_MOUSE_R_BUTTON_DOWN = WM_RBUTTONDOWN, // right button down
		SG_MOUSE_M_BUTTON_DOWN = WM_MBUTTONDOWN, // middle button down

		SG_MOUSE_L_BUTTON_UP = WM_LBUTTONUP, // left button up
		SG_MOUSE_R_BUTTON_UP = WM_RBUTTONUP, // right button up
		SG_MOUSE_M_BUTTON_UP = WM_MBUTTONUP, // middle button up

		SG_MOUSE_L_BUTTON_DOUBLE_CLICK = WM_LBUTTONDBLCLK, // left button doble click
		SG_MOUSE_R_BUTTON_DOUBLE_CLICK = WM_RBUTTONDBLCLK, // right button double click
		SG_MOUSE_M_BUTTON_DOUBLE_CLICK = WM_MBUTTONDBLCLK, // middle button double click
	};

	enum SG_PAD
	{
		// Number pad
		SG_PAD_LOCK = VK_NUMLOCK, // NumLock
		SG_PAD_N0   = VK_NUMPAD0, // pad 0
		SG_PAD_N9  = VK_NUMPAD9, // pad 9
		SG_PAD_N8  = VK_NUMPAD8, // pad 8
		SG_PAD_N7  = VK_NUMPAD7, // pad 7
		SG_PAD_N6  = VK_NUMPAD6, // pad 6
		SG_PAD_N5  = VK_NUMPAD5, // pad 5
		SG_PAD_N4  = VK_NUMPAD4, // pad 4
		SG_PAD_N3  = VK_NUMPAD3, // pad 3
		SG_PAD_N2  = VK_NUMPAD2, // pad 2
		SG_PAD_N1  = VK_NUMPAD1, // pad 1
	};
	
	struct SG_FUNCTIONS_HOLDER
	{
		void  (*hCreateFunc)(void);
		void  (*hReshapeFunc)(unsigned width, unsigned height);
		void  (*hKeyboardFunc)(SG_KEYS keyboard, SG_KEY_STATUS keystatus);
		void  (*hMouseFunc)(SG_MOUSE mouse, unsigned xpos, unsigned ypos);
		void  (*hDisplayFunc)(void);
	};

//////////////////////////////////////////////////////////////////////////////////////////////

// Create MFC window
_DLL void OnCreate(HWND hWnd, HDC *hDC, HGLRC *hRC);
// Destroy MFC window
_DLL void OnDestroy(HDC hDC, HGLRC hRC);
// 當窗口大小發生改動時，對窗口重新繪製
_DLL void OnReshapeWindow(LPARAM lParam, void (*func)(unsigned width, unsigned height));
// 對鍵盤事件進行反應
_DLL void OnKeyboardEvents(
	UINT msg, WPARAM wParam, void (*func)(SG_KEYS keyboard, SG_KEY_STATUS keystatus));
// 對鼠標事件進行反應
_DLL void OnMouseEvents(
	UINT msg, LPARAM lParam, void (*func)(SG_MOUSE mouse, unsigned xpos, unsigned ypos));
// 繪製內容
_DLL void OnDisplay(void (*func)(void));

};

#endif