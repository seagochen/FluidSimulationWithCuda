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
#define DENSITY             33.7f
#define VELOCITY            40.5f
#define STEPSIZE           0.001f

#define TIMES                 80

#define VOLUME_X             128
#define VOLUME_Y             128
#define VOLUME_Z             128

#define THREADS_S           1024

#define WINDOWS_X            400
#define WINDOWS_Y            400

#define NODES_X                4
#define NODES_Y                4
#define NODES_Z                4

#define GGRIDS_X              64
#define GGRIDS_Y              64
#define GGRIDS_Z              64

#define GBULLET_X             66
#define GBULLET_Y             66
#define GBULLET_Z             66

#define NTILE_X               32
#define NTILE_Y               32

#define GBTILE_X              22
#define GBTILE_Y              22

#define SGRIDS_X              32
#define SGRIDS_Y              32
#define SGRIDS_Z              32

#define SBULLET_X             34
#define SBULLET_Y             34
#define SBULLET_Z             34

#define SBTITLE_X             34
#define SBTITLE_Y             17

#define STANDARD_S             5
#define ENTIRE_S              11

#define CANVAS_X             400
#define CANVAS_Y             400

#define TPBUFFER_S          1024

#define APP_TITLE   "Refinement x128 FPS: %d "

#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

#endif