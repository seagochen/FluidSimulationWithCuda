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

/* ��CopyBuffer��C���Է�װ������������ݿ�������ʱ������ */
extern void hostCopyBuffer( double *buff, const sge::SGSTDGRID *grids, const sge::SGFIELDTYPE type );

/* ��CopyBuffer��C���Է�װ������ʱ���ݿ����������� */
extern void hostCopyBuffer( sge::SGSTDGRID *grids, const double *buff, const sge::SGFIELDTYPE type );

/* ��SwapBuffer��C���Է�װ����������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
extern void hostSwapBuffer( double *buf1, double *buf2 );

/* ��ZeroBuffer��C���Է�װ����GPU buffer������������ */
extern void hostZeroBuffer( int nPtrs, ... );

/* ����������м������� */
extern void AddSource( double *buffer, sge::SGSTDGRID *grids, sge::SGFIELDTYPE type );

#endif