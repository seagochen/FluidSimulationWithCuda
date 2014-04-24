/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Apr 03, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME            0.5f
#define DIFFUSION            0.1f
#define VISOCITY             0.0f
#define DENSITY             33.7f
#define VELOCITY            40.5f

#define GRIDS_X              128
#define GRIDS_Y              128
#define GRIDS_Z              128

#define BULLET_X             130
#define BULLET_Y             130
#define BULLET_Z             130

#define STEPSIZE           0.001f

#define VOLUME_X             128
#define VOLUME_Y             128
#define VOLUME_Z             128

#define THREADS_S           1024

#define TILE_X                32
#define TILE_Y                32

#define WINDOWS_X            400
#define WINDOWS_Y            400

#define CANVAS_X             400
#define CANVAS_Y             400

#define TPBUFFER_S          1024

#define APP_TITLE   "Host x128 FPS: %d "

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

#define TIMES                50

#endif