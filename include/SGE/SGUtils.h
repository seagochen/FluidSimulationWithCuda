/**
* ------------------------------------the MIT License--------------------------------------------------
*
* Copyright (C) year 2013 Orlando Chen <seagochen@gmail.com>
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
* <File>        SGUtils.h
* <Specification>
*	Main header file of SGE
*/

#ifndef _SEAGOSOFT_SGUTILS_H_
#define _SEAGOSOFT_SGUTILS_H_

#if defined(_SEAGOSOFT_AUXILIARY_H_)
#error Do not need include Auxiliaries.h when you already included SGUtils.h
#endif
#if defined(_SEAGOSOFT_FILE_MANAGER_H_)
#error Do not need include Filemanager.h when you already included SGUtils.h
#endif
#if defined(_SEAGOSOFT_ENGINE_GAMEEVENT_H_)
#error Do not need include GameEvents.h when you already included SGUtils.h
#endif
#if defined(_SEAGOSOFT_ENGINE_MAINACTIVITY_H_)
#error Do not need include MainActivity.h when you already included SGUtils.h
#endif
#if defined(_SEAGOSOFT_FREETYPE_H_)
#error Do not need include FreeType.h when you already included SGUtils.h
#endif
#if defined(_INCLUDES_NECESSARY_LIBS_)
#error Do not need load those libraries when you already included SGUtils.h
#endif

#include "Auxiliaries.h"
#include "FileManager.h"
#include "GameEvents.h"
#include "MainActivity.h"
#include "FreeType.h"

#ifndef _INCLUDES_NECESSARY_LIBS_
#define _INCLUDES_NECESSARY_LIBS_
#pragma once
#pragma comment (lib, "sgeglt.lib")
#pragma comment (lib, "sgeutil.lib")
#pragma comment (lib, "glew32.lib")
#pragma comment (lib, "freeglut.lib")
#pragma comment (lib, "GlU32.Lib")
#endif

#endif