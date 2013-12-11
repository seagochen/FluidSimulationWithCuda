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
* <First>       Dec 12, 2013
* <Last>		Dec 12, 2013
* <File>        visualAux.cpp
*/

#ifndef __visual_aux_cpp_
#define __visual_aux_cpp_

#include <iostream>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GLT/glm/glm.hpp>
#include <GLT/glm/gtc/matrix_transform.hpp>
#include <GLT/glm/gtx/transform2.hpp>
#include <GLT/glm/gtc/type_ptr.hpp>

#include "cfdHeader.h"
#include "GLSLHelper.h"


/*
-----------------------------------------------------------------------------------------------------------
* @function CoutFPS
* @author   Orlando Chen
* @date     Nov 6, 2013
* @input    NULL
* @return   NULL
* @bref     When flush the image onto screen, call this function to count FPS and update its status      
-----------------------------------------------------------------------------------------------------------
*/
void CountFPS ( void ) 
{
	// Calculate the number of frames per one second:
	m_fps->dwFrames++;
	m_fps->dwCurrentTime = GetTickCount(); // Even better to use timeGetTime()
	m_fps->dwElapsedTime = m_fps->dwCurrentTime - m_fps->dwLastUpdateTime;
	
	// Already 1s
	if ( m_fps->dwElapsedTime >= 1000 )
	{
		m_fps->FPS = m_fps->dwFrames * 1000.0 / m_fps->dwElapsedTime;
		m_fps->dwFrames = 0;
		m_fps->dwLastUpdateTime = m_fps->dwCurrentTime;
	}

//	glPushMatrix();
//	{
//		glLoadIdentity();									// Reset The Current Modelview Matrix
//		glTranslatef(0.0f,0.0f,-1.0f);						// Move One Unit Into The Screen
//		
//		// White Text
//		glColor3f(0.0f, 1.0f, 0.0f);
//		m_font->EnableFreeType();
//		m_font->PrintText(*m_font, 10, 10, "Current's FPS:   %d", m_fps->FPS);
//		m_font->DisableFreeType();
//	}
//	glPopMatrix();
	
	system ("cls");
	printf ("Current's FPS:   %d", m_fps->FPS);
}

#endif