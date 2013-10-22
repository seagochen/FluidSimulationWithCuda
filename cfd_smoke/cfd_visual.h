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
* <First>       Sep 13, 2013
* <Last>		Oct 22, 2013
* <File>        cfd_visual.h
*/

#ifndef _VISUALIZATION_H_
#define _VISUALIZATION_H_

#include <GL\glew.h>
#include <GL\freeglut.h>
#include <SGE\SGUtils.h>

//////////////////////////////////////////////////////////////////////////////////////////
///

/*
  ----------------------------------------------------------------------
   Definitions of Structure
  ----------------------------------------------------------------------
*/

namespace sge
{
	struct _mouse
	{
		unsigned pre_cursor_x, pre_cursor_y;
		unsigned cur_cursor_x, cur_cursor_y;
		bool left_button_pressed;
		bool right_button_pressed;
	};

	struct _fps
	{
		DWORD dwFrames;
		DWORD dwCurrentTime;
		DWORD dwLastUpdateTime;
		DWORD dwElapsedTime;
		UINT  FPS;
	};

	struct _volume3D
	{
		GLubyte *data;
		unsigned texture_id;
		int width, height, depth;
		unsigned size;
	};

	struct _volume2D
	{
		GLubyte *data;
		unsigned texture_id;
		int width, height;
		unsigned size;
	};

	struct _viewMatrix
	{
		GLfloat view_angle;
		GLfloat z_near, z_far;
		GLfloat eye_x, eye_y, eye_z;
		GLfloat look_x, look_y, look_z;
		GLfloat up_x, up_y, up_z;
		GLfloat z_forward;
	};

};


/*
  ----------------------------------------------------------------------
   Definition of Visual
  ----------------------------------------------------------------------
*/

namespace sge
{
	typedef class Visual
	{
	public:
		Visual(GLuint width, GLuint height, MainActivity *hActivity);
		~Visual(void);

	public:
		static void OnCreate   (void);
		static void OnResize   (void);
		static void OnIdle     (void);
		static void OnDisplay  (void);
		static void OnKeyboard (SG_KEYS keys, SG_KEY_STATUS status);
		static void OnMouse    (SG_MOUSE mouse, GLuint x_pos, GLuint y_pos);
		static void OnDestroy  (void);
	}Visualization;
};

///
//////////////////////////////////////////////////////////////////////////////////////////

#endif