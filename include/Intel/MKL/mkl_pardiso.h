/*******************************************************************************
*   Copyright(C) 2004-2013 Intel Corporation. All Rights Reserved.
*
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*
*   *Third Party trademarks are the property of their respective owners.
*
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************
*   Content : MKL PARDISO C header file
*
*           Contains interface to PARDISO.
*
********************************************************************************
*/
#if !defined( __MKL_PARDISO_H )

#define __MKL_PARDISO_H

#include "mkl_dss.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#if !defined(MKL_CALL_CONV)

   #if !defined(__MIC__)
      #if defined(MKL_STDCALL)
         #define MKL_CALL_CONV __stdcall
      #else
         #define MKL_CALL_CONV __cdecl
      #endif
   #else
      #define MKL_CALL_CONV
   #endif
#endif

#if  !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)   extern rtype MKL_CALL_CONV   name    arg;
#endif

_Mkl_Api(void,pardiso,(
    _MKL_DSS_HANDLE_t,   MKL_INT *,     MKL_INT *,
    MKL_INT *,           MKL_INT *,     MKL_INT *,
    void *,              MKL_INT *,     MKL_INT *,
    MKL_INT *,           MKL_INT *,     MKL_INT *,
    MKL_INT *,           void *, void *,
   MKL_INT *))

_Mkl_Api(void,PARDISO,(
    _MKL_DSS_HANDLE_t,   MKL_INT *,     MKL_INT *,
    MKL_INT *,           MKL_INT *,     MKL_INT *,
    void *,              MKL_INT *,     MKL_INT *,
    MKL_INT *,           MKL_INT *,     MKL_INT *,
    MKL_INT *,           void *, void *,
   MKL_INT *))

_Mkl_Api(void,pardisoinit,(
    _MKL_DSS_HANDLE_t,   MKL_INT *,  MKL_INT *))

_Mkl_Api(void,PARDISOINIT,(
    _MKL_DSS_HANDLE_t,   MKL_INT *,   MKL_INT *))

_Mkl_Api(void,pardiso_64,(
    _MKL_DSS_HANDLE_t,     long long int *,     long long int *,
    long long int *,       long long int *,     long long int *,
    void *,                long long int *,     long long int *,
    long long int *,       long long int *,     long long int *,
    long long int *,       void *,              void *,
        long long int *))
_Mkl_Api(void,PARDISO_64,(
    _MKL_DSS_HANDLE_t,     long long int *,     long long int *,
    long long int *,       long long int *,     long long int *,
    void *,                long long int *,     long long int *,
    long long int *,       long long int *,     long long int *,
    long long int *,       void *,              void *,
        long long int *))
enum PARDISO_ENV_PARAM {
       PARDISO_OOC_FILE_NAME = 1
};

/* Error classes */
#define PARDISO_NO_ERROR                 0
#define PARDISO_UNIMPLEMENTED         -101
#define PARDISO_NULL_HANDLE           -102
#define PARDISO_MEMORY_ERROR          -103

_Mkl_Api(MKL_INT,pardiso_getenv,(const _MKL_DSS_HANDLE_t, const enum PARDISO_ENV_PARAM*, char*))
_Mkl_Api(MKL_INT,PARDISO_GETENV,(const _MKL_DSS_HANDLE_t, const enum PARDISO_ENV_PARAM*, char*))

_Mkl_Api(MKL_INT,pardiso_setenv,(_MKL_DSS_HANDLE_t, const enum PARDISO_ENV_PARAM*, const char*))
_Mkl_Api(MKL_INT,PARDISO_SETENV,(_MKL_DSS_HANDLE_t, const enum PARDISO_ENV_PARAM*, const char*))

/* MKL Progress routine */
#ifndef _MKL_PARDISO_PIVOT_H_
#define _MKL_PARDISO_PIVOT_H_
_Mkl_Api(int,MKL_PARDISO_PIVOT,( double* aii, double* bii, double* eps ))
_Mkl_Api(int,MKL_PARDISO_PIVOT_,( double* aii, double* bii, double* eps ))
_Mkl_Api(int,mkl_pardiso_pivot,( double* aii, double* bii, double* eps ))
_Mkl_Api(int,mkl_pardiso_pivot_,( double* aii, double* bii, double* eps ))
#endif /* _MKL_PARDISO_PIVOT_H_ */
_Mkl_Api(void,pardiso_getdiag,( _MKL_DSS_HANDLE_t, void *,       void *, MKL_INT *, MKL_INT *  ))

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
