/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 27, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME            0.5f
#define DIFFUSION            0.1f
#define VISOCITY             0.0f
#define DENSITY             12.5f
#define VELOCITY            15.7f

#define TOY_X                 32
#define TOY_Y                 32
#define TOY_Z                 32

#define GRIDS_X               64
#define GRIDS_Y               64
#define GRIDS_Z               64

#define BULLET_X              66
#define BULLET_Y              66
#define BULLET_Z              66

#define NODES_X                2
#define NODES_Y                2
#define NODES_Z                2

#define STEPSIZE           0.001f

#define VOLUME_X              64
#define VOLUME_Y              64
#define VOLUME_Z              64

#define THREADS_S           1024

#define TILE_X                32
#define TILE_Y                32

#define WINDOWS_X            400
#define WINDOWS_Y            400

#define CANVAS_X             400
#define CANVAS_Y             400

#define TPBUFFER_S          1024

#define APP_TITLE   "Prototype v2.00.00. FPS: %d "

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

#endif