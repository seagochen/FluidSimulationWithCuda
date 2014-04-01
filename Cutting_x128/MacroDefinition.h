/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Apr 01, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME            0.5f
#define DIFFUSION            0.1f
#define VISOCITY             0.0f
#define DENSITY             33.7f
#define VELOCITY            40.5f
#define STEPSIZE           0.001f

#define TIMES                 80
#define THREADS_S           1024

#define VOLUME_X             128
#define VOLUME_Y             128
#define VOLUME_Z             128

#define GLOBAL_X              64
#define GLOBAL_Y              64
#define GLOBAL_Z              64
#define GLOBAL_BX             66
#define GLOBAL_BY             66
#define GLOBAL_BZ             66

#define EXTEND_X             128
#define EXTEND_Y             128
#define EXTEND_Z             128

#define SUBNODE_X             32
#define SUBNODE_Y             32
#define SUBNODE_Z             32
#define SUBNODE_BX            34
#define SUBNODE_BY            34
#define SUBNODE_BZ            34

#define WINDOWS_X            400
#define WINDOWS_Y            400

#define CANVAS_X             400
#define CANVAS_Y             400

#define APP_TITLE   "Cutting x128 FPS: %d "

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

#endif