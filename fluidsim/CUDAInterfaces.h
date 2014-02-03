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

/* 对CopyBuffer的C语言封装，将网格的数据拷贝到临时数据中 */
extern void hostCopyBuffer( double *buff, const sge::SGSTDGRID *grids, const sge::SGFIELDTYPE type );

/* 对CopyBuffer的C语言封装，将临时数据拷贝到网格中 */
extern void hostCopyBuffer( sge::SGSTDGRID *grids, const double *buff, const sge::SGFIELDTYPE type );

/* 对SwapBuffer的C语言封装，交换两段GPU buffer的数据，需要注意的是这两段数据的长度应该是一样的，是64^3 */
extern void hostSwapBuffer( double *buf1, double *buf2 );

/* 对ZeroBuffer的C语言封装，对GPU buffer的数据做归零 */
extern void hostZeroBuffer( int nPtrs, ... );

/* 向计算网格中加入数据 */
extern void AddSource( double *buffer, sge::SGSTDGRID *grids, sge::SGFIELDTYPE type );

#endif