/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 30, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME            0.5f
#define DIFFUSION            0.1f
#define VISOCITY             0.0f
#define DENSITY             12.5f
#define VELOCITY            15.7f
#define STEPSIZE           0.001f

#define TIMES                 80

#define gGRIDS_X              64
#define gGRIDS_Y              64
#define gGRIDS_Z              64

#define gBULLET_X             66
#define gBULLET_Y             66
#define gBULLET_Z             66

#define sNODE_X                2
#define sNODE_Y                2
#define sNODE_Z                2

#define sGRIDS_X             128
#define sGRIDS_Y             128
#define sGRIDS_Z             128

#define sBULLET_X            130
#define sBULLET_Y            130
#define sBULLET_Z            130

#define BIG_X                256
#define BIG_Y                256
#define BIG_Z                256

#define VOLUME_X             256
#define VOLUME_Y             256
#define VOLUME_Z             256

#define THREADS_S           1024

#define TILE_X                32
#define TILE_Y                32

#define WINDOWS_X            400
#define WINDOWS_Y            400

#define CANVAS_X             400
#define CANVAS_Y             400

#define TPBUFFER_S          1024

#define APP_TITLE   "Refinement x256 FPS: %d "

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

#endif