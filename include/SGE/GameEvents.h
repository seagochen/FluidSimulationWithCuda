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

namespace sge {

	enum SGKEYS
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
		SG_KEY_LEFT       = VK_LEFT,  // Left Arrow
		SG_KEY_ARRUP      = VK_UP,    // Up Arrow
		SG_KEY_ARRRIGHT   = VK_RIGHT, // Right Arrow
		SG_KEY_ARRDOWN    = VK_DOWN,  // Down Arrow
		
		//定义数据字符0~9 
		SG_KEY_0 = 0x30,   // num 0
		SG_KEY_1 = 0x31,   // num 1
		SG_KEY_2 = 0x32,   // num 2
		SG_KEY_3 = 0x33,   // num 3
		SG_KEY_4 = 0x34,   // num 4
		SG_KEY_5 = 0x35,   // num 5
		SG_KEY_6 = 0x36,   // num 6
		SG_KEY_7 = 0x37,   // num 7
		SG_KEY_8 = 0x38,   // num 8
		SG_KEY_9 = 0x39,   // num 9
		
		//定义数据字符A~Z
		SG_KEY_A = 0x41,   // A
		SG_KEY_B = 0x42,   // B
		SG_KEY_C = 0x43,   // C
		SG_KEY_D = 0x44,   // D
		SG_KEY_E = 0x45,   // E
		SG_KEY_F = 0x46,   // F
		SG_KEY_G = 0x47,   // G
		SG_KEY_H = 0x48,   // H
		SG_KEY_I = 0x49,   // I
		SG_KEY_J = 0x4A,   // J
		SG_KEY_K = 0x4B,   // K
		SG_KEY_L = 0x4C,   // L
		SG_KEY_M = 0x4D,   // M
		SG_KEY_N = 0x4E,   // N
		SG_KEY_O = 0x4F,   // O
		SG_KEY_P = 0x50,   // P
		SG_KEY_Q = 0x51,   // Q
		SG_KEY_R = 0x52,   // R
		SG_KEY_S = 0x53,   // S
		SG_KEY_T = 0x54,   // T
		SG_KEY_U = 0x55,   // U
		SG_KEY_V = 0x56,   // V
		SG_KEY_W = 0x57,   // W
		SG_KEY_X = 0x58,   // X
		SG_KEY_Y = 0x59,   // Y
		SG_KEY_Z = 0x5A,   // Z
		
		//定义数据字符a~z  
		SG_KEY_a = 0x61,   // a
		SG_KEY_b = 0x62,   // b
		SG_KEY_c = 0x63,   // c
		SG_KEY_d = 0x64,   // d
		SG_KEY_e = 0x65,   // e
		SG_KEY_f = 0x66,   // f
		SG_KEY_g = 0x67,   // g
		SG_KEY_h = 0x68,   // h
		SG_KEY_i = 0x69,   // i
		SG_KEY_j = 0x6A,   // j
		SG_KEY_k = 0x6B,   // k
		SG_KEY_l = 0x6C,   // l
		SG_KEY_m = 0x6D,   // m
		SG_KEY_n = 0x6E,   // n
		SG_KEY_o = 0x6F,   // o
		SG_KEY_p = 0x70,   // p
		SG_KEY_q = 0x71,   // k
		SG_KEY_r = 0x72,   // r
		SG_KEY_s = 0x73,   // s
		SG_KEY_t = 0x74,   // t
		SG_KEY_u = 0x75,   // u
		SG_KEY_v = 0x76,   // v
		SG_KEY_w = 0x77,   // w
		SG_KEY_x = 0x78,   // x
		SG_KEY_y = 0x79,   // y
		SG_KEY_z = 0x7A,   // z
		
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

	enum SGMOUSE
	{
		SG_MOUSE_L = VK_LBUTTON,  // 鼠标左键
		SG_MOUSE_R = VK_RBUTTON,  // 鼠标右键
		SG_MOUSE_M = VK_MBUTTON,  // 鼠标中键
	};

	enum SGPAD
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

//////////////////////////////////////////////////////////////////////////////////////////////
	
#include "Auxiliaries.h"

#include <Windows.h>
#include <GL\glut.h>

// Create MFC window
DllExport void OnCreate(HWND hWnd, HDC *hDC, HGLRC *hRC);
// Destroy MFC window
DllExport void OnDestroy(HDC hDC, HGLRC hRC);
// 當窗口大小發生改動時，對窗口重新繪製
DllExport void OnReshapeWindow(LPARAM lParam, void (*func)(int, int));
// 對指定的窗口進行繪圖工作的準備
DllExport void OnPaint(HWND hWnd);
// 對鍵盤事件進行反應
DllExport void OnKeyDown(void (*func)(SGKEYS));
// 對鍵盤事件進行反應
DllExport void OnKeyUp(void (*func)(SGKEYS));
// 對鼠標事件進行反應
DllExport void OnMouseLeftDown();
// 對鼠標事件進行反應
DllExport void OnMouseLeftUp();
// 繪製內容
DllExport void OnDisplay(void (*func)(void));

};

#endif