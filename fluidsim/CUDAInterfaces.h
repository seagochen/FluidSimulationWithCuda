/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 02, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     CUDAInterfaces.h
*/

#ifndef __cuda_interfaces_h_
#define __cuda_interfaces_h_

#include "DataStructures.h"

using namespace sge;

/* 对CopyBuffer的C语言封装，将网格的数据拷贝到临时数据中 */
extern void hostCopyBuffer( SGTEMPBUFFERS *buffs, const SGCUDANODES *nodes, const SGFIELDTYPE type );

/* 对CopyBuffer的C语言封装，将临时数据拷贝到网格中 */
extern void hostCopyBuffer( SGCUDANODES *nodes, const SGTEMPBUFFERS *buffs, const SGFIELDTYPE type );

/* 对SwapBuffer的C语言封装，交换两段GPU buffer的数据，需要注意的是这两段数据的长度应该是一样的，是64^3 */
extern void hostSwapBuffer( SGTEMPBUFFERS *buf1, SGTEMPBUFFERS *buf2 );

/* 对ZeroBuffer的C语言封装，对GPU buffer的数据做归零 */
extern void hostZeroBuffer( SGTEMPBUFFERS *buf );

/* 采集网格数据，并转换为volumetric data */
extern void hostPickData ( unsigned char *data, const SGTEMPBUFFERS *bufs,
	int const offi, int const offj, int const offk );

/* 向计算网格中加入数据 */
extern void hostAdvection( double *stores, SGTEMPBUFFERS *buff, SGCUDANODES *nodes );

/* 求解密度场 */
extern void DensitySolver
	( SGTEMPBUFFERS *dens, SGTEMPBUFFERS *dens0, SGCUDANODES *nodes, double *stores );

/* 求解速度场 */
extern void hostVelocitySolver
	( SGTEMPBUFFERS *u, SGTEMPBUFFERS *v, SGTEMPBUFFERS *w, SGTEMPBUFFERS *div, SGTEMPBUFFERS *p,
	SGTEMPBUFFERS *u0, SGTEMPBUFFERS *v0, SGTEMPBUFFERS *w0,
	SGCUDANODES *nodes, double *stores );

#endif