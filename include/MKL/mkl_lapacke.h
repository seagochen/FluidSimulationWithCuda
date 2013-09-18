/*****************************************************************************
* Copyright(C) 2010-2013 Intel Corporation. All Rights Reserved.
*
* The source code, information  and  material ("Material") contained herein is
* owned  by Intel Corporation or its suppliers or licensors, and title to such
* Material remains  with Intel Corporation  or its suppliers or licensors. The
* Material  contains proprietary information  of  Intel or  its  suppliers and
* licensors. The  Material is protected by worldwide copyright laws and treaty
* provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
* modified, published, uploaded, posted, transmitted, distributed or disclosed
* in any way  without Intel's  prior  express written  permission. No  license
* under  any patent, copyright  or  other intellectual property rights  in the
* Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
* implication, inducement,  estoppel or  otherwise.  Any  license  under  such
* intellectual  property  rights must  be express  and  approved  by  Intel in
* writing.
*
* *Third Party trademarks are the property of their respective owners.
*
* Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
* this  notice or  any other notice embedded  in Materials by Intel or Intel's
* suppliers or licensors in any way.
*
******************************************************************************
* Contents: Native C interface to LAPACK
* Author: Intel Corporation
* Generated January, 2010
*****************************************************************************/

#ifndef _MKL_LAPACKE_H_
#define _MKL_LAPACKE_H_

#include "mkl_types.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define LAPACK_ROW_MAJOR               101
#define LAPACK_COL_MAJOR               102

#define LAPACK_WORK_MEMORY_ERROR       -1010
#define LAPACK_TRANSPOSE_MEMORY_ERROR  -1011

#ifndef lapack_int
#define lapack_int              MKL_INT
#endif

#ifndef lapack_logical
#define lapack_logical          lapack_int
#endif

/* Complex types are structures equivalent to the
* Fortran complex types COMPLEX(4) and COMPLEX(8).
*
* One can also redefine the types with his own types
* for example by including in the code definitions like
*
* #define lapack_complex_float std::complex<float>
* #define lapack_complex_double std::complex<double>
*
* or define these types in the command line:
*
* -Dlapack_complex_float="std::complex<float>"
* -Dlapack_complex_double="std::complex<double>"
*/

#ifndef lapack_complex_float
#define lapack_complex_float    MKL_Complex8
#endif

#ifndef lapack_complex_double
#define lapack_complex_double   MKL_Complex16
#endif

/* Callback logical functions of one, two, or three arguments are used
*  to select eigenvalues to sort to the top left of the Schur form.
*  The value is selected if function returns non-zero. */

typedef lapack_logical (*LAPACK_S_SELECT2) ( const float*, const float* );
typedef lapack_logical (*LAPACK_S_SELECT3)
    ( const float*, const float*, const float* );
typedef lapack_logical (*LAPACK_D_SELECT2) ( const double*, const double* );
typedef lapack_logical (*LAPACK_D_SELECT3)
    ( const double*, const double*, const double* );

typedef lapack_logical (*LAPACK_C_SELECT1) ( const lapack_complex_float* );
typedef lapack_logical (*LAPACK_C_SELECT2)
    ( const lapack_complex_float*, const lapack_complex_float* );
typedef lapack_logical (*LAPACK_Z_SELECT1) ( const lapack_complex_double* );
typedef lapack_logical (*LAPACK_Z_SELECT2)
    ( const lapack_complex_double*, const lapack_complex_double* );

lapack_int LAPACKE_slasrt( char id, lapack_int n, float* d);
lapack_int LAPACKE_dlasrt( char id, lapack_int n, double* d);

lapack_int LAPACKE_sbbcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, float* theta, float* phi,
                           float* u1, lapack_int ldu1, float* u2,
                           lapack_int ldu2, float* v1t, lapack_int ldv1t,
                           float* v2t, lapack_int ldv2t, float* b11d,
                           float* b11e, float* b12d, float* b12e, float* b21d,
                           float* b21e, float* b22d, float* b22e );
lapack_int LAPACKE_dbbcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, double* theta,
                           double* phi, double* u1, lapack_int ldu1, double* u2,
                           lapack_int ldu2, double* v1t, lapack_int ldv1t,
                           double* v2t, lapack_int ldv2t, double* b11d,
                           double* b11e, double* b12d, double* b12e,
                           double* b21d, double* b21e, double* b22d,
                           double* b22e );
lapack_int LAPACKE_cbbcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, float* theta, float* phi,
                           lapack_complex_float* u1, lapack_int ldu1,
                           lapack_complex_float* u2, lapack_int ldu2,
                           lapack_complex_float* v1t, lapack_int ldv1t,
                           lapack_complex_float* v2t, lapack_int ldv2t,
                           float* b11d, float* b11e, float* b12d, float* b12e,
                           float* b21d, float* b21e, float* b22d, float* b22e );
lapack_int LAPACKE_zbbcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, lapack_int m,
                           lapack_int p, lapack_int q, double* theta,
                           double* phi, lapack_complex_double* u1,
                           lapack_int ldu1, lapack_complex_double* u2,
                           lapack_int ldu2, lapack_complex_double* v1t,
                           lapack_int ldv1t, lapack_complex_double* v2t,
                           lapack_int ldv2t, double* b11d, double* b11e,
                           double* b12d, double* b12e, double* b21d,
                           double* b21e, double* b22d, double* b22e );

lapack_int LAPACKE_sbdsdc( int matrix_order, char uplo, char compq,
                           lapack_int n, float* d, float* e, float* u,
                           lapack_int ldu, float* vt, lapack_int ldvt, float* q,
                           lapack_int* iq );
lapack_int LAPACKE_dbdsdc( int matrix_order, char uplo, char compq,
                           lapack_int n, double* d, double* e, double* u,
                           lapack_int ldu, double* vt, lapack_int ldvt,
                           double* q, lapack_int* iq );

lapack_int LAPACKE_sbdsqr( int matrix_order, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           float* d, float* e, float* vt, lapack_int ldvt,
                           float* u, lapack_int ldu, float* c, lapack_int ldc );
lapack_int LAPACKE_dbdsqr( int matrix_order, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           double* d, double* e, double* vt, lapack_int ldvt,
                           double* u, lapack_int ldu, double* c,
                           lapack_int ldc );
lapack_int LAPACKE_cbdsqr( int matrix_order, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           float* d, float* e, lapack_complex_float* vt,
                           lapack_int ldvt, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* c,
                           lapack_int ldc );
lapack_int LAPACKE_zbdsqr( int matrix_order, char uplo, lapack_int n,
                           lapack_int ncvt, lapack_int nru, lapack_int ncc,
                           double* d, double* e, lapack_complex_double* vt,
                           lapack_int ldvt, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* c,
                           lapack_int ldc );

lapack_int LAPACKE_sdisna( char job, lapack_int m, lapack_int n, const float* d,
                           float* sep );
lapack_int LAPACKE_ddisna( char job, lapack_int m, lapack_int n,
                           const double* d, double* sep );

lapack_int LAPACKE_sgbbrd( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, float* ab, lapack_int ldab, float* d,
                           float* e, float* q, lapack_int ldq, float* pt,
                           lapack_int ldpt, float* c, lapack_int ldc );
lapack_int LAPACKE_dgbbrd( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, double* ab, lapack_int ldab,
                           double* d, double* e, double* q, lapack_int ldq,
                           double* pt, lapack_int ldpt, double* c,
                           lapack_int ldc );
lapack_int LAPACKE_cgbbrd( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, lapack_complex_float* ab,
                           lapack_int ldab, float* d, float* e,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_complex_float* pt, lapack_int ldpt,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zgbbrd( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int ncc, lapack_int kl,
                           lapack_int ku, lapack_complex_double* ab,
                           lapack_int ldab, double* d, double* e,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* pt, lapack_int ldpt,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_sgbcon( int matrix_order, char norm, lapack_int n,
                           lapack_int kl, lapack_int ku, const float* ab,
                           lapack_int ldab, const lapack_int* ipiv, float anorm,
                           float* rcond );
lapack_int LAPACKE_dgbcon( int matrix_order, char norm, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* ab,
                           lapack_int ldab, const lapack_int* ipiv,
                           double anorm, double* rcond );
lapack_int LAPACKE_cgbcon( int matrix_order, char norm, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zgbcon( int matrix_order, char norm, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_sgbequ( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const float* ab,
                           lapack_int ldab, float* r, float* c, float* rowcnd,
                           float* colcnd, float* amax );
lapack_int LAPACKE_dgbequ( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, const double* ab,
                           lapack_int ldab, double* r, double* c,
                           double* rowcnd, double* colcnd, double* amax );
lapack_int LAPACKE_cgbequ( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           const lapack_complex_float* ab, lapack_int ldab,
                           float* r, float* c, float* rowcnd, float* colcnd,
                           float* amax );
lapack_int LAPACKE_zgbequ( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           const lapack_complex_double* ab, lapack_int ldab,
                           double* r, double* c, double* rowcnd, double* colcnd,
                           double* amax );

lapack_int LAPACKE_sgbequb( int matrix_order, lapack_int m, lapack_int n,
                            lapack_int kl, lapack_int ku, const float* ab,
                            lapack_int ldab, float* r, float* c, float* rowcnd,
                            float* colcnd, float* amax );
lapack_int LAPACKE_dgbequb( int matrix_order, lapack_int m, lapack_int n,
                            lapack_int kl, lapack_int ku, const double* ab,
                            lapack_int ldab, double* r, double* c,
                            double* rowcnd, double* colcnd, double* amax );
lapack_int LAPACKE_cgbequb( int matrix_order, lapack_int m, lapack_int n,
                            lapack_int kl, lapack_int ku,
                            const lapack_complex_float* ab, lapack_int ldab,
                            float* r, float* c, float* rowcnd, float* colcnd,
                            float* amax );
lapack_int LAPACKE_zgbequb( int matrix_order, lapack_int m, lapack_int n,
                            lapack_int kl, lapack_int ku,
                            const lapack_complex_double* ab, lapack_int ldab,
                            double* r, double* c, double* rowcnd,
                            double* colcnd, double* amax );

lapack_int LAPACKE_sgbrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const float* ab, lapack_int ldab, const float* afb,
                           lapack_int ldafb, const lapack_int* ipiv,
                           const float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dgbrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const double* ab, lapack_int ldab, const double* afb,
                           lapack_int ldafb, const lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_cgbrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_complex_float* afb, lapack_int ldafb,
                           const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zgbrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_complex_double* afb, lapack_int ldafb,
                           const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sgbrfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, const float* ab, lapack_int ldab,
                            const float* afb, lapack_int ldafb,
                            const lapack_int* ipiv, const float* r,
                            const float* c, const float* b, lapack_int ldb,
                            float* x, lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dgbrfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, const double* ab, lapack_int ldab,
                            const double* afb, lapack_int ldafb,
                            const lapack_int* ipiv, const double* r,
                            const double* c, const double* b, lapack_int ldb,
                            double* x, lapack_int ldx, double* rcond,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );
lapack_int LAPACKE_cgbrfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, const lapack_complex_float* ab,
                            lapack_int ldab, const lapack_complex_float* afb,
                            lapack_int ldafb, const lapack_int* ipiv,
                            const float* r, const float* c,
                            const lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_zgbrfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, const lapack_complex_double* ab,
                            lapack_int ldab, const lapack_complex_double* afb,
                            lapack_int ldafb, const lapack_int* ipiv,
                            const double* r, const double* c,
                            const lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );

lapack_int LAPACKE_sgbsv( int matrix_order, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs, float* ab,
                          lapack_int ldab, lapack_int* ipiv, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgbsv( int matrix_order, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs, double* ab,
                          lapack_int ldab, lapack_int* ipiv, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cgbsv( int matrix_order, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgbsv( int matrix_order, lapack_int n, lapack_int kl,
                          lapack_int ku, lapack_int nrhs,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );

lapack_int LAPACKE_sgbsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int kl, lapack_int ku,
                           lapack_int nrhs, float* ab, lapack_int ldab,
                           float* afb, lapack_int ldafb, lapack_int* ipiv,
                           char* equed, float* r, float* c, float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr,
                           float* rpivot );
lapack_int LAPACKE_dgbsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int kl, lapack_int ku,
                           lapack_int nrhs, double* ab, lapack_int ldab,
                           double* afb, lapack_int ldafb, lapack_int* ipiv,
                           char* equed, double* r, double* c, double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr,
                           double* rpivot );
lapack_int LAPACKE_cgbsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int kl, lapack_int ku,
                           lapack_int nrhs, lapack_complex_float* ab,
                           lapack_int ldab, lapack_complex_float* afb,
                           lapack_int ldafb, lapack_int* ipiv, char* equed,
                           float* r, float* c, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr, float* rpivot );
lapack_int LAPACKE_zgbsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int kl, lapack_int ku,
                           lapack_int nrhs, lapack_complex_double* ab,
                           lapack_int ldab, lapack_complex_double* afb,
                           lapack_int ldafb, lapack_int* ipiv, char* equed,
                           double* r, double* c, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, double* rcond, double* ferr,
                           double* berr, double* rpivot );

lapack_int LAPACKE_sgbsvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, float* ab, lapack_int ldab,
                            float* afb, lapack_int ldafb, lapack_int* ipiv,
                            char* equed, float* r, float* c, float* b,
                            lapack_int ldb, float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dgbsvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, double* ab, lapack_int ldab,
                            double* afb, lapack_int ldafb, lapack_int* ipiv,
                            char* equed, double* r, double* c, double* b,
                            lapack_int ldb, double* x, lapack_int ldx,
                            double* rcond, double* rpvgrw, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );
lapack_int LAPACKE_cgbsvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, lapack_complex_float* ab,
                            lapack_int ldab, lapack_complex_float* afb,
                            lapack_int ldafb, lapack_int* ipiv, char* equed,
                            float* r, float* c, lapack_complex_float* b,
                            lapack_int ldb, lapack_complex_float* x,
                            lapack_int ldx, float* rcond, float* rpvgrw,
                            float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_zgbsvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int kl, lapack_int ku,
                            lapack_int nrhs, lapack_complex_double* ab,
                            lapack_int ldab, lapack_complex_double* afb,
                            lapack_int ldafb, lapack_int* ipiv, char* equed,
                            double* r, double* c, lapack_complex_double* b,
                            lapack_int ldb, lapack_complex_double* x,
                            lapack_int ldx, double* rcond, double* rpvgrw,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );

lapack_int LAPACKE_sgbtrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, float* ab,
                           lapack_int ldab, lapack_int* ipiv );
lapack_int LAPACKE_dgbtrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku, double* ab,
                           lapack_int ldab, lapack_int* ipiv );
lapack_int LAPACKE_cgbtrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_int* ipiv );
lapack_int LAPACKE_zgbtrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int kl, lapack_int ku,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_int* ipiv );

lapack_int LAPACKE_sgbtrs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const float* ab, lapack_int ldab,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dgbtrs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const double* ab, lapack_int ldab,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_cgbtrs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zgbtrs( int matrix_order, char trans, lapack_int n,
                           lapack_int kl, lapack_int ku, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_sgebak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const float* scale,
                           lapack_int m, float* v, lapack_int ldv );
lapack_int LAPACKE_dgebak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const double* scale,
                           lapack_int m, double* v, lapack_int ldv );
lapack_int LAPACKE_cgebak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const float* scale,
                           lapack_int m, lapack_complex_float* v,
                           lapack_int ldv );
lapack_int LAPACKE_zgebak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const double* scale,
                           lapack_int m, lapack_complex_double* v,
                           lapack_int ldv );

lapack_int LAPACKE_sgebal( int matrix_order, char job, lapack_int n, float* a,
                           lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                           float* scale );
lapack_int LAPACKE_dgebal( int matrix_order, char job, lapack_int n, double* a,
                           lapack_int lda, lapack_int* ilo, lapack_int* ihi,
                           double* scale );
lapack_int LAPACKE_cgebal( int matrix_order, char job, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ilo, lapack_int* ihi, float* scale );
lapack_int LAPACKE_zgebal( int matrix_order, char job, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ilo, lapack_int* ihi, double* scale );

lapack_int LAPACKE_sgebrd( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* d, float* e,
                           float* tauq, float* taup );
lapack_int LAPACKE_dgebrd( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* d, double* e,
                           double* tauq, double* taup );
lapack_int LAPACKE_cgebrd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* d,
                           float* e, lapack_complex_float* tauq,
                           lapack_complex_float* taup );
lapack_int LAPACKE_zgebrd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda, double* d,
                           double* e, lapack_complex_double* tauq,
                           lapack_complex_double* taup );

lapack_int LAPACKE_sgecon( int matrix_order, char norm, lapack_int n,
                           const float* a, lapack_int lda, float anorm,
                           float* rcond );
lapack_int LAPACKE_dgecon( int matrix_order, char norm, lapack_int n,
                           const double* a, lapack_int lda, double anorm,
                           double* rcond );
lapack_int LAPACKE_cgecon( int matrix_order, char norm, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           float anorm, float* rcond );
lapack_int LAPACKE_zgecon( int matrix_order, char norm, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           double anorm, double* rcond );

lapack_int LAPACKE_sgeequ( int matrix_order, lapack_int m, lapack_int n,
                           const float* a, lapack_int lda, float* r, float* c,
                           float* rowcnd, float* colcnd, float* amax );
lapack_int LAPACKE_dgeequ( int matrix_order, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda, double* r,
                           double* c, double* rowcnd, double* colcnd,
                           double* amax );
lapack_int LAPACKE_cgeequ( int matrix_order, lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           float* r, float* c, float* rowcnd, float* colcnd,
                           float* amax );
lapack_int LAPACKE_zgeequ( int matrix_order, lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           double* r, double* c, double* rowcnd, double* colcnd,
                           double* amax );

lapack_int LAPACKE_sgeequb( int matrix_order, lapack_int m, lapack_int n,
                            const float* a, lapack_int lda, float* r, float* c,
                            float* rowcnd, float* colcnd, float* amax );
lapack_int LAPACKE_dgeequb( int matrix_order, lapack_int m, lapack_int n,
                            const double* a, lapack_int lda, double* r,
                            double* c, double* rowcnd, double* colcnd,
                            double* amax );
lapack_int LAPACKE_cgeequb( int matrix_order, lapack_int m, lapack_int n,
                            const lapack_complex_float* a, lapack_int lda,
                            float* r, float* c, float* rowcnd, float* colcnd,
                            float* amax );
lapack_int LAPACKE_zgeequb( int matrix_order, lapack_int m, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            double* r, double* c, double* rowcnd,
                            double* colcnd, double* amax );

lapack_int LAPACKE_sgees( int matrix_order, char jobvs, char sort,
                          LAPACK_S_SELECT2 select, lapack_int n, float* a,
                          lapack_int lda, lapack_int* sdim, float* wr,
                          float* wi, float* vs, lapack_int ldvs );
lapack_int LAPACKE_dgees( int matrix_order, char jobvs, char sort,
                          LAPACK_D_SELECT2 select, lapack_int n, double* a,
                          lapack_int lda, lapack_int* sdim, double* wr,
                          double* wi, double* vs, lapack_int ldvs );
lapack_int LAPACKE_cgees( int matrix_order, char jobvs, char sort,
                          LAPACK_C_SELECT1 select, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_float* w,
                          lapack_complex_float* vs, lapack_int ldvs );
lapack_int LAPACKE_zgees( int matrix_order, char jobvs, char sort,
                          LAPACK_Z_SELECT1 select, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* sdim, lapack_complex_double* w,
                          lapack_complex_double* vs, lapack_int ldvs );

lapack_int LAPACKE_sgeesx( int matrix_order, char jobvs, char sort,
                           LAPACK_S_SELECT2 select, char sense, lapack_int n,
                           float* a, lapack_int lda, lapack_int* sdim,
                           float* wr, float* wi, float* vs, lapack_int ldvs,
                           float* rconde, float* rcondv );
lapack_int LAPACKE_dgeesx( int matrix_order, char jobvs, char sort,
                           LAPACK_D_SELECT2 select, char sense, lapack_int n,
                           double* a, lapack_int lda, lapack_int* sdim,
                           double* wr, double* wi, double* vs, lapack_int ldvs,
                           double* rconde, double* rcondv );
lapack_int LAPACKE_cgeesx( int matrix_order, char jobvs, char sort,
                           LAPACK_C_SELECT1 select, char sense, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* sdim, lapack_complex_float* w,
                           lapack_complex_float* vs, lapack_int ldvs,
                           float* rconde, float* rcondv );
lapack_int LAPACKE_zgeesx( int matrix_order, char jobvs, char sort,
                           LAPACK_Z_SELECT1 select, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* sdim, lapack_complex_double* w,
                           lapack_complex_double* vs, lapack_int ldvs,
                           double* rconde, double* rcondv );

lapack_int LAPACKE_sgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* wr,
                          float* wi, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_dgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* wr,
                          double* wi, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_cgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* w, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_zgeev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* w,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr );

lapack_int LAPACKE_sgeevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, float* a,
                           lapack_int lda, float* wr, float* wi, float* vl,
                           lapack_int ldvl, float* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, float* scale,
                           float* abnrm, float* rconde, float* rcondv );
lapack_int LAPACKE_dgeevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, double* a,
                           lapack_int lda, double* wr, double* wi, double* vl,
                           lapack_int ldvl, double* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, double* scale,
                           double* abnrm, double* rconde, double* rcondv );
lapack_int LAPACKE_cgeevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* w, lapack_complex_float* vl,
                           lapack_int ldvl, lapack_complex_float* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           float* scale, float* abnrm, float* rconde,
                           float* rcondv );
lapack_int LAPACKE_zgeevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* w, lapack_complex_double* vl,
                           lapack_int ldvl, lapack_complex_double* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           double* scale, double* abnrm, double* rconde,
                           double* rcondv );

lapack_int LAPACKE_sgehrd( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           float* tau );
lapack_int LAPACKE_dgehrd( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           double* tau );
lapack_int LAPACKE_cgehrd( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* tau );
lapack_int LAPACKE_zgehrd( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* tau );

lapack_int LAPACKE_sgejsv( int matrix_order, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, float* a, lapack_int lda, float* sva,
                           float* u, lapack_int ldu, float* v, lapack_int ldv,
                           float* stat, lapack_int* istat );
lapack_int LAPACKE_dgejsv( int matrix_order, char joba, char jobu, char jobv,
                           char jobr, char jobt, char jobp, lapack_int m,
                           lapack_int n, double* a, lapack_int lda, double* sva,
                           double* u, lapack_int ldu, double* v, lapack_int ldv,
                           double* stat, lapack_int* istat );

lapack_int LAPACKE_sgelqf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dgelqf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_cgelqf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zgelqf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_sgels( int matrix_order, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, float* a,
                          lapack_int lda, float* b, lapack_int ldb );
lapack_int LAPACKE_dgels( int matrix_order, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs, double* a,
                          lapack_int lda, double* b, lapack_int ldb );
lapack_int LAPACKE_cgels( int matrix_order, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zgels( int matrix_order, char trans, lapack_int m,
                          lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_sgelsd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_dgelsd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
lapack_int LAPACKE_cgelsd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_zgelsd( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_sgelss( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_dgelss( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );
lapack_int LAPACKE_cgelss( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* s, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_zgelss( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* s, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_sgelsy( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_dgelsy( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, lapack_int* jpvt,
                           double rcond, lapack_int* rank );
lapack_int LAPACKE_cgelsy( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_int* jpvt, float rcond,
                           lapack_int* rank );
lapack_int LAPACKE_zgelsy( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* jpvt, double rcond,
                           lapack_int* rank );

lapack_int LAPACKE_sgemqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const float* v, lapack_int ldv,
                            const float* t, lapack_int ldt, float* c,
                            lapack_int ldc );
lapack_int LAPACKE_dgemqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const double* v, lapack_int ldv,
                            const double* t, lapack_int ldt, double* c,
                            lapack_int ldc );
lapack_int LAPACKE_cgemqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const lapack_complex_float* v,
                            lapack_int ldv, const lapack_complex_float* t,
                            lapack_int ldt, lapack_complex_float* c,
                            lapack_int ldc );
lapack_int LAPACKE_zgemqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k,
                            lapack_int nb, const lapack_complex_double* v,
                            lapack_int ldv, const lapack_complex_double* t,
                            lapack_int ldt, lapack_complex_double* c,
                            lapack_int ldc );

lapack_int LAPACKE_sgeqlf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dgeqlf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_cgeqlf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zgeqlf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_sgeqp3( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* jpvt,
                           float* tau );
lapack_int LAPACKE_dgeqp3( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* jpvt,
                           double* tau );
lapack_int LAPACKE_cgeqp3( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_float* tau );
lapack_int LAPACKE_zgeqp3( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_double* tau );

lapack_int LAPACKE_sgeqpf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* jpvt,
                           float* tau );
lapack_int LAPACKE_dgeqpf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* jpvt,
                           double* tau );
lapack_int LAPACKE_cgeqpf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_float* tau );
lapack_int LAPACKE_zgeqpf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* jpvt, lapack_complex_double* tau );

lapack_int LAPACKE_sgeqrf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dgeqrf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_cgeqrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zgeqrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_sgeqrfp( int matrix_order, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dgeqrfp( int matrix_order, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_cgeqrfp( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* tau );
lapack_int LAPACKE_zgeqrfp( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* tau );

lapack_int LAPACKE_sgeqrt( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nb, float* a, lapack_int lda, float* t,
                           lapack_int ldt );
lapack_int LAPACKE_dgeqrt( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nb, double* a, lapack_int lda, double* t,
                           lapack_int ldt );
lapack_int LAPACKE_cgeqrt( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nb, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* t,
                           lapack_int ldt );
lapack_int LAPACKE_zgeqrt( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int nb, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* t,
                           lapack_int ldt );

lapack_int LAPACKE_sgeqrt2( int matrix_order, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* t,
                            lapack_int ldt );
lapack_int LAPACKE_dgeqrt2( int matrix_order, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* t,
                            lapack_int ldt );
lapack_int LAPACKE_cgeqrt2( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* t, lapack_int ldt );
lapack_int LAPACKE_zgeqrt2( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* t, lapack_int ldt );

lapack_int LAPACKE_sgeqrt3( int matrix_order, lapack_int m, lapack_int n,
                            float* a, lapack_int lda, float* t,
                            lapack_int ldt );
lapack_int LAPACKE_dgeqrt3( int matrix_order, lapack_int m, lapack_int n,
                            double* a, lapack_int lda, double* t,
                            lapack_int ldt );
lapack_int LAPACKE_cgeqrt3( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* t, lapack_int ldt );
lapack_int LAPACKE_zgeqrt3( int matrix_order, lapack_int m, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* t, lapack_int ldt );

lapack_int LAPACKE_sgerfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const float* af, lapack_int ldaf,
                           const lapack_int* ipiv, const float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dgerfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_cgerfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zgerfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sgerfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int nrhs, const float* a,
                            lapack_int lda, const float* af, lapack_int ldaf,
                            const lapack_int* ipiv, const float* r,
                            const float* c, const float* b, lapack_int ldb,
                            float* x, lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dgerfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int nrhs, const double* a,
                            lapack_int lda, const double* af, lapack_int ldaf,
                            const lapack_int* ipiv, const double* r,
                            const double* c, const double* b, lapack_int ldb,
                            double* x, lapack_int ldx, double* rcond,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );
lapack_int LAPACKE_cgerfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_float* a, lapack_int lda,
                            const lapack_complex_float* af, lapack_int ldaf,
                            const lapack_int* ipiv, const float* r,
                            const float* c, const lapack_complex_float* b,
                            lapack_int ldb, lapack_complex_float* x,
                            lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_zgerfsx( int matrix_order, char trans, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_double* a, lapack_int lda,
                            const lapack_complex_double* af, lapack_int ldaf,
                            const lapack_int* ipiv, const double* r,
                            const double* c, const lapack_complex_double* b,
                            lapack_int ldb, lapack_complex_double* x,
                            lapack_int ldx, double* rcond, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );

lapack_int LAPACKE_sgerqf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dgerqf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_cgerqf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zgerqf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_sgesdd( int matrix_order, char jobz, lapack_int m,
                           lapack_int n, float* a, lapack_int lda, float* s,
                           float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_dgesdd( int matrix_order, char jobz, lapack_int m,
                           lapack_int n, double* a, lapack_int lda, double* s,
                           double* u, lapack_int ldu, double* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_cgesdd( int matrix_order, char jobz, lapack_int m,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt );
lapack_int LAPACKE_zgesdd( int matrix_order, char jobz, lapack_int m,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt );

lapack_int LAPACKE_sgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                          float* a, lapack_int lda, lapack_int* ipiv, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                          double* a, lapack_int lda, lapack_int* ipiv,
                          double* b, lapack_int ldb );
lapack_int LAPACKE_cgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );
lapack_int LAPACKE_dsgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                           double* a, lapack_int lda, lapack_int* ipiv,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           lapack_int* iter );
lapack_int LAPACKE_zcgesv( int matrix_order, lapack_int n, lapack_int nrhs,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* iter );

lapack_int LAPACKE_sgesvd( int matrix_order, char jobu, char jobvt,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* s, float* u, lapack_int ldu, float* vt,
                           lapack_int ldvt, float* superb );
lapack_int LAPACKE_dgesvd( int matrix_order, char jobu, char jobvt,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* s, double* u, lapack_int ldu,
                           double* vt, lapack_int ldvt, double* superb );
lapack_int LAPACKE_cgesvd( int matrix_order, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float* s, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* vt,
                           lapack_int ldvt, float* superb );
lapack_int LAPACKE_zgesvd( int matrix_order, char jobu, char jobvt,
                           lapack_int m, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double* s, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* vt,
                           lapack_int ldvt, double* superb );

lapack_int LAPACKE_sgesvj( int matrix_order, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, float* a, lapack_int lda,
                           float* sva, lapack_int mv, float* v, lapack_int ldv,
                           float* stat );
lapack_int LAPACKE_dgesvj( int matrix_order, char joba, char jobu, char jobv,
                           lapack_int m, lapack_int n, double* a,
                           lapack_int lda, double* sva, lapack_int mv,
                           double* v, lapack_int ldv, double* stat );

lapack_int LAPACKE_sgesvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs, float* a,
                           lapack_int lda, float* af, lapack_int ldaf,
                           lapack_int* ipiv, char* equed, float* r, float* c,
                           float* b, lapack_int ldb, float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr,
                           float* rpivot );
lapack_int LAPACKE_dgesvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs, double* a,
                           lapack_int lda, double* af, lapack_int ldaf,
                           lapack_int* ipiv, char* equed, double* r, double* c,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr,
                           double* rpivot );
lapack_int LAPACKE_cgesvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* af, lapack_int ldaf,
                           lapack_int* ipiv, char* equed, float* r, float* c,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr,
                           float* rpivot );
lapack_int LAPACKE_zgesvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* af, lapack_int ldaf,
                           lapack_int* ipiv, char* equed, double* r, double* c,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr,
                           double* rpivot );

lapack_int LAPACKE_sgesvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int nrhs, float* a,
                            lapack_int lda, float* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, float* r, float* c,
                            float* b, lapack_int ldb, float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dgesvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int nrhs, double* a,
                            lapack_int lda, double* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, double* r, double* c,
                            double* b, lapack_int ldb, double* x,
                            lapack_int ldx, double* rcond, double* rpvgrw,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );
lapack_int LAPACKE_cgesvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, float* r, float* c,
                            lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_zgesvxx( int matrix_order, char fact, char trans,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, double* r, double* c,
                            lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* rpvgrw, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );

lapack_int LAPACKE_sgetrf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dgetrf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_cgetrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zgetrf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );

lapack_int LAPACKE_sgetri( int matrix_order, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dgetri( int matrix_order, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_cgetri( int matrix_order, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zgetri( int matrix_order, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );

lapack_int LAPACKE_sgetrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dgetrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_cgetrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zgetrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_sggbak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const float* lscale,
                           const float* rscale, lapack_int m, float* v,
                           lapack_int ldv );
lapack_int LAPACKE_dggbak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const double* lscale,
                           const double* rscale, lapack_int m, double* v,
                           lapack_int ldv );
lapack_int LAPACKE_cggbak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const float* lscale,
                           const float* rscale, lapack_int m,
                           lapack_complex_float* v, lapack_int ldv );
lapack_int LAPACKE_zggbak( int matrix_order, char job, char side, lapack_int n,
                           lapack_int ilo, lapack_int ihi, const double* lscale,
                           const double* rscale, lapack_int m,
                           lapack_complex_double* v, lapack_int ldv );

lapack_int LAPACKE_sggbal( int matrix_order, char job, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb,
                           lapack_int* ilo, lapack_int* ihi, float* lscale,
                           float* rscale );
lapack_int LAPACKE_dggbal( int matrix_order, char job, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           lapack_int* ilo, lapack_int* ihi, double* lscale,
                           double* rscale );
lapack_int LAPACKE_cggbal( int matrix_order, char job, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_int* ilo, lapack_int* ihi, float* lscale,
                           float* rscale );
lapack_int LAPACKE_zggbal( int matrix_order, char job, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_int* ilo, lapack_int* ihi, double* lscale,
                           double* rscale );

lapack_int LAPACKE_sgges( int matrix_order, char jobvsl, char jobvsr, char sort,
                          LAPACK_S_SELECT3 selctg, lapack_int n, float* a,
                          lapack_int lda, float* b, lapack_int ldb,
                          lapack_int* sdim, float* alphar, float* alphai,
                          float* beta, float* vsl, lapack_int ldvsl, float* vsr,
                          lapack_int ldvsr );
lapack_int LAPACKE_dgges( int matrix_order, char jobvsl, char jobvsr, char sort,
                          LAPACK_D_SELECT3 selctg, lapack_int n, double* a,
                          lapack_int lda, double* b, lapack_int ldb,
                          lapack_int* sdim, double* alphar, double* alphai,
                          double* beta, double* vsl, lapack_int ldvsl,
                          double* vsr, lapack_int ldvsr );
lapack_int LAPACKE_cgges( int matrix_order, char jobvsl, char jobvsr, char sort,
                          LAPACK_C_SELECT2 selctg, lapack_int n,
                          lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb,
                          lapack_int* sdim, lapack_complex_float* alpha,
                          lapack_complex_float* beta, lapack_complex_float* vsl,
                          lapack_int ldvsl, lapack_complex_float* vsr,
                          lapack_int ldvsr );
lapack_int LAPACKE_zgges( int matrix_order, char jobvsl, char jobvsr, char sort,
                          LAPACK_Z_SELECT2 selctg, lapack_int n,
                          lapack_complex_double* a, lapack_int lda,
                          lapack_complex_double* b, lapack_int ldb,
                          lapack_int* sdim, lapack_complex_double* alpha,
                          lapack_complex_double* beta,
                          lapack_complex_double* vsl, lapack_int ldvsl,
                          lapack_complex_double* vsr, lapack_int ldvsr );

lapack_int LAPACKE_sggesx( int matrix_order, char jobvsl, char jobvsr,
                           char sort, LAPACK_S_SELECT3 selctg, char sense,
                           lapack_int n, float* a, lapack_int lda, float* b,
                           lapack_int ldb, lapack_int* sdim, float* alphar,
                           float* alphai, float* beta, float* vsl,
                           lapack_int ldvsl, float* vsr, lapack_int ldvsr,
                           float* rconde, float* rcondv );
lapack_int LAPACKE_dggesx( int matrix_order, char jobvsl, char jobvsr,
                           char sort, LAPACK_D_SELECT3 selctg, char sense,
                           lapack_int n, double* a, lapack_int lda, double* b,
                           lapack_int ldb, lapack_int* sdim, double* alphar,
                           double* alphai, double* beta, double* vsl,
                           lapack_int ldvsl, double* vsr, lapack_int ldvsr,
                           double* rconde, double* rcondv );
lapack_int LAPACKE_cggesx( int matrix_order, char jobvsl, char jobvsr,
                           char sort, LAPACK_C_SELECT2 selctg, char sense,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_int* sdim,
                           lapack_complex_float* alpha,
                           lapack_complex_float* beta,
                           lapack_complex_float* vsl, lapack_int ldvsl,
                           lapack_complex_float* vsr, lapack_int ldvsr,
                           float* rconde, float* rcondv );
lapack_int LAPACKE_zggesx( int matrix_order, char jobvsl, char jobvsr,
                           char sort, LAPACK_Z_SELECT2 selctg, char sense,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_int* sdim,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vsl, lapack_int ldvsl,
                           lapack_complex_double* vsr, lapack_int ldvsr,
                           double* rconde, double* rcondv );

lapack_int LAPACKE_sggev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, float* a, lapack_int lda, float* b,
                          lapack_int ldb, float* alphar, float* alphai,
                          float* beta, float* vl, lapack_int ldvl, float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_dggev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, double* a, lapack_int lda, double* b,
                          lapack_int ldb, double* alphar, double* alphai,
                          double* beta, double* vl, lapack_int ldvl, double* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_cggev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_float* a, lapack_int lda,
                          lapack_complex_float* b, lapack_int ldb,
                          lapack_complex_float* alpha,
                          lapack_complex_float* beta, lapack_complex_float* vl,
                          lapack_int ldvl, lapack_complex_float* vr,
                          lapack_int ldvr );
lapack_int LAPACKE_zggev( int matrix_order, char jobvl, char jobvr,
                          lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb, lapack_complex_double* alpha,
                          lapack_complex_double* beta,
                          lapack_complex_double* vl, lapack_int ldvl,
                          lapack_complex_double* vr, lapack_int ldvr );

lapack_int LAPACKE_sggevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb,
                           float* alphar, float* alphai, float* beta, float* vl,
                           lapack_int ldvl, float* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, float* lscale,
                           float* rscale, float* abnrm, float* bbnrm,
                           float* rconde, float* rcondv );
lapack_int LAPACKE_dggevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double* alphar, double* alphai, double* beta,
                           double* vl, lapack_int ldvl, double* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           double* lscale, double* rscale, double* abnrm,
                           double* bbnrm, double* rconde, double* rcondv );
lapack_int LAPACKE_cggevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* alpha,
                           lapack_complex_float* beta, lapack_complex_float* vl,
                           lapack_int ldvl, lapack_complex_float* vr,
                           lapack_int ldvr, lapack_int* ilo, lapack_int* ihi,
                           float* lscale, float* rscale, float* abnrm,
                           float* bbnrm, float* rconde, float* rcondv );
lapack_int LAPACKE_zggevx( int matrix_order, char balanc, char jobvl,
                           char jobvr, char sense, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* vl, lapack_int ldvl,
                           lapack_complex_double* vr, lapack_int ldvr,
                           lapack_int* ilo, lapack_int* ihi, double* lscale,
                           double* rscale, double* abnrm, double* bbnrm,
                           double* rconde, double* rcondv );

lapack_int LAPACKE_sggglm( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* d, float* x, float* y );
lapack_int LAPACKE_dggglm( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, double* a, lapack_int lda, double* b,
                           lapack_int ldb, double* d, double* x, double* y );
lapack_int LAPACKE_cggglm( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* d,
                           lapack_complex_float* x, lapack_complex_float* y );
lapack_int LAPACKE_zggglm( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* d,
                           lapack_complex_double* x, lapack_complex_double* y );

lapack_int LAPACKE_sgghrd( int matrix_order, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           float* a, lapack_int lda, float* b, lapack_int ldb,
                           float* q, lapack_int ldq, float* z, lapack_int ldz );
lapack_int LAPACKE_dgghrd( int matrix_order, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           double* a, lapack_int lda, double* b, lapack_int ldb,
                           double* q, lapack_int ldq, double* z,
                           lapack_int ldz );
lapack_int LAPACKE_cgghrd( int matrix_order, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zgghrd( int matrix_order, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_sgglse( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int p, float* a, lapack_int lda, float* b,
                           lapack_int ldb, float* c, float* d, float* x );
lapack_int LAPACKE_dgglse( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int p, double* a, lapack_int lda, double* b,
                           lapack_int ldb, double* c, double* d, double* x );
lapack_int LAPACKE_cgglse( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* c,
                           lapack_complex_float* d, lapack_complex_float* x );
lapack_int LAPACKE_zgglse( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* c,
                           lapack_complex_double* d, lapack_complex_double* x );

lapack_int LAPACKE_sggqrf( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, float* a, lapack_int lda, float* taua,
                           float* b, lapack_int ldb, float* taub );
lapack_int LAPACKE_dggqrf( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, double* a, lapack_int lda,
                           double* taua, double* b, lapack_int ldb,
                           double* taub );
lapack_int LAPACKE_cggqrf( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* taua,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* taub );
lapack_int LAPACKE_zggqrf( int matrix_order, lapack_int n, lapack_int m,
                           lapack_int p, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* taua,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* taub );

lapack_int LAPACKE_sggrqf( int matrix_order, lapack_int m, lapack_int p,
                           lapack_int n, float* a, lapack_int lda, float* taua,
                           float* b, lapack_int ldb, float* taub );
lapack_int LAPACKE_dggrqf( int matrix_order, lapack_int m, lapack_int p,
                           lapack_int n, double* a, lapack_int lda,
                           double* taua, double* b, lapack_int ldb,
                           double* taub );
lapack_int LAPACKE_cggrqf( int matrix_order, lapack_int m, lapack_int p,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* taua,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* taub );
lapack_int LAPACKE_zggrqf( int matrix_order, lapack_int m, lapack_int p,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* taua,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* taub );

lapack_int LAPACKE_sggsvd( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l, float* a,
                           lapack_int lda, float* b, lapack_int ldb,
                           float* alpha, float* beta, float* u, lapack_int ldu,
                           float* v, lapack_int ldv, float* q, lapack_int ldq,
                           lapack_int* iwork );
lapack_int LAPACKE_dggsvd( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double* alpha, double* beta, double* u,
                           lapack_int ldu, double* v, lapack_int ldv, double* q,
                           lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_cggsvd( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           float* alpha, float* beta, lapack_complex_float* u,
                           lapack_int ldu, lapack_complex_float* v,
                           lapack_int ldv, lapack_complex_float* q,
                           lapack_int ldq, lapack_int* iwork );
lapack_int LAPACKE_zggsvd( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int n, lapack_int p,
                           lapack_int* k, lapack_int* l,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           double* alpha, double* beta,
                           lapack_complex_double* u, lapack_int ldu,
                           lapack_complex_double* v, lapack_int ldv,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_int* iwork );

lapack_int LAPACKE_sggsvp( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float tola,
                           float tolb, lapack_int* k, lapack_int* l, float* u,
                           lapack_int ldu, float* v, lapack_int ldv, float* q,
                           lapack_int ldq );
lapack_int LAPACKE_dggsvp( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double tola, double tolb, lapack_int* k,
                           lapack_int* l, double* u, lapack_int ldu, double* v,
                           lapack_int ldv, double* q, lapack_int ldq );
lapack_int LAPACKE_cggsvp( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb, float tola,
                           float tolb, lapack_int* k, lapack_int* l,
                           lapack_complex_float* u, lapack_int ldu,
                           lapack_complex_float* v, lapack_int ldv,
                           lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zggsvp( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           double tola, double tolb, lapack_int* k,
                           lapack_int* l, lapack_complex_double* u,
                           lapack_int ldu, lapack_complex_double* v,
                           lapack_int ldv, lapack_complex_double* q,
                           lapack_int ldq );

lapack_int LAPACKE_sgtcon( char norm, lapack_int n, const float* dl,
                           const float* d, const float* du, const float* du2,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_dgtcon( char norm, lapack_int n, const double* dl,
                           const double* d, const double* du, const double* du2,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
lapack_int LAPACKE_cgtcon( char norm, lapack_int n,
                           const lapack_complex_float* dl,
                           const lapack_complex_float* d,
                           const lapack_complex_float* du,
                           const lapack_complex_float* du2,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zgtcon( char norm, lapack_int n,
                           const lapack_complex_double* dl,
                           const lapack_complex_double* d,
                           const lapack_complex_double* du,
                           const lapack_complex_double* du2,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_sgtrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const float* dl, const float* d,
                           const float* du, const float* dlf, const float* df,
                           const float* duf, const float* du2,
                           const lapack_int* ipiv, const float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dgtrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const double* dl, const double* d,
                           const double* du, const double* dlf,
                           const double* df, const double* duf,
                           const double* du2, const lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_cgtrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* dl,
                           const lapack_complex_float* d,
                           const lapack_complex_float* du,
                           const lapack_complex_float* dlf,
                           const lapack_complex_float* df,
                           const lapack_complex_float* duf,
                           const lapack_complex_float* du2,
                           const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zgtrfs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* dl,
                           const lapack_complex_double* d,
                           const lapack_complex_double* du,
                           const lapack_complex_double* dlf,
                           const lapack_complex_double* df,
                           const lapack_complex_double* duf,
                           const lapack_complex_double* du2,
                           const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sgtsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          float* dl, float* d, float* du, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dgtsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          double* dl, double* d, double* du, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cgtsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          lapack_complex_float* dl, lapack_complex_float* d,
                          lapack_complex_float* du, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zgtsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          lapack_complex_double* dl, lapack_complex_double* d,
                          lapack_complex_double* du, lapack_complex_double* b,
                          lapack_int ldb );

lapack_int LAPACKE_sgtsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs, const float* dl,
                           const float* d, const float* du, float* dlf,
                           float* df, float* duf, float* du2, lapack_int* ipiv,
                           const float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dgtsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs, const double* dl,
                           const double* d, const double* du, double* dlf,
                           double* df, double* duf, double* du2,
                           lapack_int* ipiv, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
lapack_int LAPACKE_cgtsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* dl,
                           const lapack_complex_float* d,
                           const lapack_complex_float* du,
                           lapack_complex_float* dlf, lapack_complex_float* df,
                           lapack_complex_float* duf, lapack_complex_float* du2,
                           lapack_int* ipiv, const lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_zgtsvx( int matrix_order, char fact, char trans,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* dl,
                           const lapack_complex_double* d,
                           const lapack_complex_double* du,
                           lapack_complex_double* dlf,
                           lapack_complex_double* df,
                           lapack_complex_double* duf,
                           lapack_complex_double* du2, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_sgttrf( lapack_int n, float* dl, float* d, float* du,
                           float* du2, lapack_int* ipiv );
lapack_int LAPACKE_dgttrf( lapack_int n, double* dl, double* d, double* du,
                           double* du2, lapack_int* ipiv );
lapack_int LAPACKE_cgttrf( lapack_int n, lapack_complex_float* dl,
                           lapack_complex_float* d, lapack_complex_float* du,
                           lapack_complex_float* du2, lapack_int* ipiv );
lapack_int LAPACKE_zgttrf( lapack_int n, lapack_complex_double* dl,
                           lapack_complex_double* d, lapack_complex_double* du,
                           lapack_complex_double* du2, lapack_int* ipiv );

lapack_int LAPACKE_sgttrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const float* dl, const float* d,
                           const float* du, const float* du2,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dgttrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const double* dl, const double* d,
                           const double* du, const double* du2,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_cgttrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* dl,
                           const lapack_complex_float* d,
                           const lapack_complex_float* du,
                           const lapack_complex_float* du2,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zgttrs( int matrix_order, char trans, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* dl,
                           const lapack_complex_double* d,
                           const lapack_complex_double* du,
                           const lapack_complex_double* du2,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_chbev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int kd, lapack_complex_float* ab,
                          lapack_int ldab, float* w, lapack_complex_float* z,
                          lapack_int ldz );
lapack_int LAPACKE_zhbev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int kd, lapack_complex_double* ab,
                          lapack_int ldab, double* w, lapack_complex_double* z,
                          lapack_int ldz );

lapack_int LAPACKE_chbevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_float* ab,
                           lapack_int ldab, float* w, lapack_complex_float* z,
                           lapack_int ldz );
lapack_int LAPACKE_zhbevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_double* ab,
                           lapack_int ldab, double* w, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_chbevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhbevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* q, lapack_int ldq, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_chbgst( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_float* ab, lapack_int ldab,
                           const lapack_complex_float* bb, lapack_int ldbb,
                           lapack_complex_float* x, lapack_int ldx );
lapack_int LAPACKE_zhbgst( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_double* ab, lapack_int ldab,
                           const lapack_complex_double* bb, lapack_int ldbb,
                           lapack_complex_double* x, lapack_int ldx );

lapack_int LAPACKE_chbgv( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_complex_float* bb, lapack_int ldbb, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhbgv( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_complex_double* bb, lapack_int ldbb, double* w,
                          lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chbgvd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* bb, lapack_int ldbb, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhbgvd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* bb, lapack_int ldbb,
                           double* w, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_chbgvx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* bb, lapack_int ldbb,
                           lapack_complex_float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhbgvx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* bb, lapack_int ldbb,
                           lapack_complex_double* q, lapack_int ldq, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_chbtrd( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_float* ab,
                           lapack_int ldab, float* d, float* e,
                           lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zhbtrd( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_double* ab,
                           lapack_int ldab, double* d, double* e,
                           lapack_complex_double* q, lapack_int ldq );

lapack_int LAPACKE_checon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zhecon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_cheequb( int matrix_order, char uplo, lapack_int n,
                            const lapack_complex_float* a, lapack_int lda,
                            float* s, float* scond, float* amax );
lapack_int LAPACKE_zheequb( int matrix_order, char uplo, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            double* s, double* scond, double* amax );

lapack_int LAPACKE_cheev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* a, lapack_int lda, float* w );
lapack_int LAPACKE_zheev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* a, lapack_int lda, double* w );

lapack_int LAPACKE_cheevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* w );
lapack_int LAPACKE_zheevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           double* w );

lapack_int LAPACKE_cheevr( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_zheevr( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, lapack_complex_double* z, lapack_int ldz,
                           lapack_int* isuppz );

lapack_int LAPACKE_cheevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_zheevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_chegst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zhegst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_chegv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_float* a,
                          lapack_int lda, lapack_complex_float* b,
                          lapack_int ldb, float* w );
lapack_int LAPACKE_zhegv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb, double* w );

lapack_int LAPACKE_chegvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float* w );
lapack_int LAPACKE_zhegvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double* w );

lapack_int LAPACKE_chegvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhegvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_cherfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zherfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_cherfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_float* a, lapack_int lda,
                            const lapack_complex_float* af, lapack_int ldaf,
                            const lapack_int* ipiv, const float* s,
                            const lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_zherfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_double* a, lapack_int lda,
                            const lapack_complex_double* af, lapack_int ldaf,
                            const lapack_int* ipiv, const double* s,
                            const lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );

lapack_int LAPACKE_chesv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhesv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_chesvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* af,
                           lapack_int ldaf, lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zhesvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* af,
                           lapack_int ldaf, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_chesvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, float* s,
                            lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_zhesvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, double* s,
                            lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* rpvgrw, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );

lapack_int LAPACKE_cheswapr( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int i1,
                             lapack_int i2 );
lapack_int LAPACKE_zheswapr( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int i1,
                             lapack_int i2 );

lapack_int LAPACKE_chetrd( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda, float* d,
                           float* e, lapack_complex_float* tau );
lapack_int LAPACKE_zhetrd( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda, double* d,
                           double* e, lapack_complex_double* tau );

lapack_int LAPACKE_chetrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zhetrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );

lapack_int LAPACKE_chetri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zhetri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );

lapack_int LAPACKE_chetri2( int matrix_order, char uplo, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            const lapack_int* ipiv );
lapack_int LAPACKE_zhetri2( int matrix_order, char uplo, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            const lapack_int* ipiv );

lapack_int LAPACKE_chetri2x( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
lapack_int LAPACKE_zhetri2x( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );

lapack_int LAPACKE_chetrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhetrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_chetrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_float* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zhetrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_double* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_chfrk( int matrix_order, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, float alpha,
                          const lapack_complex_float* a, lapack_int lda,
                          float beta, lapack_complex_float* c );
lapack_int LAPACKE_zhfrk( int matrix_order, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, double alpha,
                          const lapack_complex_double* a, lapack_int lda,
                          double beta, lapack_complex_double* c );

lapack_int LAPACKE_shgeqz( int matrix_order, char job, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           float* h, lapack_int ldh, float* t, lapack_int ldt,
                           float* alphar, float* alphai, float* beta, float* q,
                           lapack_int ldq, float* z, lapack_int ldz );
lapack_int LAPACKE_dhgeqz( int matrix_order, char job, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           double* h, lapack_int ldh, double* t, lapack_int ldt,
                           double* alphar, double* alphai, double* beta,
                           double* q, lapack_int ldq, double* z,
                           lapack_int ldz );
lapack_int LAPACKE_chgeqz( int matrix_order, char job, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           lapack_complex_float* h, lapack_int ldh,
                           lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* alpha,
                           lapack_complex_float* beta, lapack_complex_float* q,
                           lapack_int ldq, lapack_complex_float* z,
                           lapack_int ldz );
lapack_int LAPACKE_zhgeqz( int matrix_order, char job, char compq, char compz,
                           lapack_int n, lapack_int ilo, lapack_int ihi,
                           lapack_complex_double* h, lapack_int ldh,
                           lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chpcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zhpcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_chpev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_complex_float* ap, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_complex_double* ap, double* w,
                          lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chpevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_complex_float* ap, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_complex_double* ap, double* w,
                           lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chpevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_float* ap, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_zhpevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_complex_double* ap, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_chpgst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_float* ap,
                           const lapack_complex_float* bp );
lapack_int LAPACKE_zhpgst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, lapack_complex_double* ap,
                           const lapack_complex_double* bp );

lapack_int LAPACKE_chpgv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_float* ap,
                          lapack_complex_float* bp, float* w,
                          lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpgv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, lapack_complex_double* ap,
                          lapack_complex_double* bp, double* w,
                          lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chpgvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_float* ap,
                           lapack_complex_float* bp, float* w,
                           lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zhpgvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, lapack_complex_double* ap,
                           lapack_complex_double* bp, double* w,
                           lapack_complex_double* z, lapack_int ldz );

lapack_int LAPACKE_chpgvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_complex_float* bp,
                           float vl, float vu, lapack_int il, lapack_int iu,
                           float abstol, lapack_int* m, float* w,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_zhpgvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_complex_double* bp,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_chprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_complex_float* afp,
                           const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zhprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_complex_double* afp,
                           const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_chpsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zhpsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );

lapack_int LAPACKE_chpsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           lapack_complex_float* afp, lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zhpsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           lapack_complex_double* afp, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_chptrd( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap, float* d, float* e,
                           lapack_complex_float* tau );
lapack_int LAPACKE_zhptrd( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap, double* d, double* e,
                           lapack_complex_double* tau );

lapack_int LAPACKE_chptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_int* ipiv );
lapack_int LAPACKE_zhptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_int* ipiv );

lapack_int LAPACKE_chptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap, const lapack_int* ipiv );
lapack_int LAPACKE_zhptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap, const lapack_int* ipiv );

lapack_int LAPACKE_chptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zhptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_shsein( int matrix_order, char job, char eigsrc, char initv,
                           lapack_logical* select, lapack_int n, const float* h,
                           lapack_int ldh, float* wr, const float* wi,
                           float* vl, lapack_int ldvl, float* vr,
                           lapack_int ldvr, lapack_int mm, lapack_int* m,
                           lapack_int* ifaill, lapack_int* ifailr );
lapack_int LAPACKE_dhsein( int matrix_order, char job, char eigsrc, char initv,
                           lapack_logical* select, lapack_int n,
                           const double* h, lapack_int ldh, double* wr,
                           const double* wi, double* vl, lapack_int ldvl,
                           double* vr, lapack_int ldvr, lapack_int mm,
                           lapack_int* m, lapack_int* ifaill,
                           lapack_int* ifailr );
lapack_int LAPACKE_chsein( int matrix_order, char job, char eigsrc, char initv,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_float* h, lapack_int ldh,
                           lapack_complex_float* w, lapack_complex_float* vl,
                           lapack_int ldvl, lapack_complex_float* vr,
                           lapack_int ldvr, lapack_int mm, lapack_int* m,
                           lapack_int* ifaill, lapack_int* ifailr );
lapack_int LAPACKE_zhsein( int matrix_order, char job, char eigsrc, char initv,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_double* h, lapack_int ldh,
                           lapack_complex_double* w, lapack_complex_double* vl,
                           lapack_int ldvl, lapack_complex_double* vr,
                           lapack_int ldvr, lapack_int mm, lapack_int* m,
                           lapack_int* ifaill, lapack_int* ifailr );

lapack_int LAPACKE_shseqr( int matrix_order, char job, char compz, lapack_int n,
                           lapack_int ilo, lapack_int ihi, float* h,
                           lapack_int ldh, float* wr, float* wi, float* z,
                           lapack_int ldz );
lapack_int LAPACKE_dhseqr( int matrix_order, char job, char compz, lapack_int n,
                           lapack_int ilo, lapack_int ihi, double* h,
                           lapack_int ldh, double* wr, double* wi, double* z,
                           lapack_int ldz );
lapack_int LAPACKE_chseqr( int matrix_order, char job, char compz, lapack_int n,
                           lapack_int ilo, lapack_int ihi,
                           lapack_complex_float* h, lapack_int ldh,
                           lapack_complex_float* w, lapack_complex_float* z,
                           lapack_int ldz );
lapack_int LAPACKE_zhseqr( int matrix_order, char job, char compz, lapack_int n,
                           lapack_int ilo, lapack_int ihi,
                           lapack_complex_double* h, lapack_int ldh,
                           lapack_complex_double* w, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_slapmr( int matrix_order, lapack_logical forwrd,
                           lapack_int m, lapack_int n, float* x, lapack_int ldx,
                           lapack_int* k );
lapack_int LAPACKE_dlapmr( int matrix_order, lapack_logical forwrd,
                           lapack_int m, lapack_int n, double* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_clapmr( int matrix_order, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_float* x,
                           lapack_int ldx, lapack_int* k );
lapack_int LAPACKE_zlapmr( int matrix_order, lapack_logical forwrd,
                           lapack_int m, lapack_int n, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* k );

lapack_int LAPACKE_slartgp( float f, float g, float* cs, float* sn, float* r );
lapack_int LAPACKE_dlartgp( double f, double g, double* cs, double* sn,
                            double* r );

lapack_int LAPACKE_slartgs( float x, float y, float sigma, float* cs,
                            float* sn );
lapack_int LAPACKE_dlartgs( double x, double y, double sigma, double* cs,
                            double* sn );

lapack_int LAPACKE_sopgtr( int matrix_order, char uplo, lapack_int n,
                           const float* ap, const float* tau, float* q,
                           lapack_int ldq );
lapack_int LAPACKE_dopgtr( int matrix_order, char uplo, lapack_int n,
                           const double* ap, const double* tau, double* q,
                           lapack_int ldq );

lapack_int LAPACKE_sopmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const float* ap,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dopmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const double* ap,
                           const double* tau, double* c, lapack_int ldc );

lapack_int LAPACKE_sorbdb( int matrix_order, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q, float* x11,
                           lapack_int ldx11, float* x12, lapack_int ldx12,
                           float* x21, lapack_int ldx21, float* x22,
                           lapack_int ldx22, float* theta, float* phi,
                           float* taup1, float* taup2, float* tauq1,
                           float* tauq2 );
lapack_int LAPACKE_dorbdb( int matrix_order, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           double* x11, lapack_int ldx11, double* x12,
                           lapack_int ldx12, double* x21, lapack_int ldx21,
                           double* x22, lapack_int ldx22, double* theta,
                           double* phi, double* taup1, double* taup2,
                           double* tauq1, double* tauq2 );

lapack_int LAPACKE_sorcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q, float* x11,
                           lapack_int ldx11, float* x12, lapack_int ldx12,
                           float* x21, lapack_int ldx21, float* x22,
                           lapack_int ldx22, float* theta, float* u1,
                           lapack_int ldu1, float* u2, lapack_int ldu2,
                           float* v1t, lapack_int ldv1t, float* v2t,
                           lapack_int ldv2t );
lapack_int LAPACKE_dorcsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           double* x11, lapack_int ldx11, double* x12,
                           lapack_int ldx12, double* x21, lapack_int ldx21,
                           double* x22, lapack_int ldx22, double* theta,
                           double* u1, lapack_int ldu1, double* u2,
                           lapack_int ldu2, double* v1t, lapack_int ldv1t,
                           double* v2t, lapack_int ldv2t );

lapack_int LAPACKE_sorgbr( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgbr( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int k, double* a,
                           lapack_int lda, const double* tau );

lapack_int LAPACKE_sorghr( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorghr( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_sorglq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorglq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_sorgql( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgql( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_sorgqr( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgqr( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_sorgrq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, float* a, lapack_int lda,
                           const float* tau );
lapack_int LAPACKE_dorgrq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, double* a, lapack_int lda,
                           const double* tau );

lapack_int LAPACKE_sorgtr( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda, const float* tau );
lapack_int LAPACKE_dorgtr( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda, const double* tau );

lapack_int LAPACKE_sormbr( int matrix_order, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormbr( int matrix_order, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

lapack_int LAPACKE_sormhr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const float* a, lapack_int lda,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dormhr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const double* a, lapack_int lda,
                           const double* tau, double* c, lapack_int ldc );

lapack_int LAPACKE_sormlq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormlq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

lapack_int LAPACKE_sormql( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormql( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

lapack_int LAPACKE_sormqr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormqr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

lapack_int LAPACKE_sormrq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const float* a, lapack_int lda, const float* tau,
                           float* c, lapack_int ldc );
lapack_int LAPACKE_dormrq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const double* a, lapack_int lda, const double* tau,
                           double* c, lapack_int ldc );

lapack_int LAPACKE_sormrz( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const float* a, lapack_int lda,
                           const float* tau, float* c, lapack_int ldc );
lapack_int LAPACKE_dormrz( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const double* a, lapack_int lda,
                           const double* tau, double* c, lapack_int ldc );

lapack_int LAPACKE_sormtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const float* a,
                           lapack_int lda, const float* tau, float* c,
                           lapack_int ldc );
lapack_int LAPACKE_dormtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n, const double* a,
                           lapack_int lda, const double* tau, double* c,
                           lapack_int ldc );

lapack_int LAPACKE_spbcon( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const float* ab, lapack_int ldab,
                           float anorm, float* rcond );
lapack_int LAPACKE_dpbcon( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const double* ab, lapack_int ldab,
                           double anorm, double* rcond );
lapack_int LAPACKE_cpbcon( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const lapack_complex_float* ab,
                           lapack_int ldab, float anorm, float* rcond );
lapack_int LAPACKE_zpbcon( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const lapack_complex_double* ab,
                           lapack_int ldab, double anorm, double* rcond );

lapack_int LAPACKE_spbequ( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const float* ab, lapack_int ldab,
                           float* s, float* scond, float* amax );
lapack_int LAPACKE_dpbequ( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const double* ab, lapack_int ldab,
                           double* s, double* scond, double* amax );
lapack_int LAPACKE_cpbequ( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const lapack_complex_float* ab,
                           lapack_int ldab, float* s, float* scond,
                           float* amax );
lapack_int LAPACKE_zpbequ( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, const lapack_complex_double* ab,
                           lapack_int ldab, double* s, double* scond,
                           double* amax );

lapack_int LAPACKE_spbrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const float* ab,
                           lapack_int ldab, const float* afb, lapack_int ldafb,
                           const float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dpbrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const double* ab,
                           lapack_int ldab, const double* afb, lapack_int ldafb,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_cpbrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_complex_float* afb, lapack_int ldafb,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zpbrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_complex_double* afb, lapack_int ldafb,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_spbstf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kb, float* bb, lapack_int ldbb );
lapack_int LAPACKE_dpbstf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kb, double* bb, lapack_int ldbb );
lapack_int LAPACKE_cpbstf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kb, lapack_complex_float* bb,
                           lapack_int ldbb );
lapack_int LAPACKE_zpbstf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kb, lapack_complex_double* bb,
                           lapack_int ldbb );

lapack_int LAPACKE_spbsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs, float* ab,
                          lapack_int ldab, float* b, lapack_int ldb );
lapack_int LAPACKE_dpbsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs, double* ab,
                          lapack_int ldab, double* b, lapack_int ldb );
lapack_int LAPACKE_cpbsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs,
                          lapack_complex_float* ab, lapack_int ldab,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpbsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int kd, lapack_int nrhs,
                          lapack_complex_double* ab, lapack_int ldab,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_spbsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, float* ab,
                           lapack_int ldab, float* afb, lapack_int ldafb,
                           char* equed, float* s, float* b, lapack_int ldb,
                           float* x, lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dpbsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, double* ab,
                           lapack_int ldab, double* afb, lapack_int ldafb,
                           char* equed, double* s, double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
lapack_int LAPACKE_cpbsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* afb, lapack_int ldafb,
                           char* equed, float* s, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_zpbsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* afb, lapack_int ldafb,
                           char* equed, double* s, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, double* rcond, double* ferr,
                           double* berr );

lapack_int LAPACKE_spbtrf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, float* ab, lapack_int ldab );
lapack_int LAPACKE_dpbtrf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, double* ab, lapack_int ldab );
lapack_int LAPACKE_cpbtrf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_float* ab,
                           lapack_int ldab );
lapack_int LAPACKE_zpbtrf( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_complex_double* ab,
                           lapack_int ldab );

lapack_int LAPACKE_spbtrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const float* ab,
                           lapack_int ldab, float* b, lapack_int ldb );
lapack_int LAPACKE_dpbtrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs, const double* ab,
                           lapack_int ldab, double* b, lapack_int ldb );
lapack_int LAPACKE_cpbtrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpbtrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_spftrf( int matrix_order, char transr, char uplo,
                           lapack_int n, float* a );
lapack_int LAPACKE_dpftrf( int matrix_order, char transr, char uplo,
                           lapack_int n, double* a );
lapack_int LAPACKE_cpftrf( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_zpftrf( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_complex_double* a );

lapack_int LAPACKE_spftri( int matrix_order, char transr, char uplo,
                           lapack_int n, float* a );
lapack_int LAPACKE_dpftri( int matrix_order, char transr, char uplo,
                           lapack_int n, double* a );
lapack_int LAPACKE_cpftri( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_zpftri( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_complex_double* a );

lapack_int LAPACKE_spftrs( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_int nrhs, const float* a,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dpftrs( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_int nrhs, const double* a,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_cpftrs( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpftrs( int matrix_order, char transr, char uplo,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_spocon( int matrix_order, char uplo, lapack_int n,
                           const float* a, lapack_int lda, float anorm,
                           float* rcond );
lapack_int LAPACKE_dpocon( int matrix_order, char uplo, lapack_int n,
                           const double* a, lapack_int lda, double anorm,
                           double* rcond );
lapack_int LAPACKE_cpocon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           float anorm, float* rcond );
lapack_int LAPACKE_zpocon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           double anorm, double* rcond );

lapack_int LAPACKE_spoequ( int matrix_order, lapack_int n, const float* a,
                           lapack_int lda, float* s, float* scond,
                           float* amax );
lapack_int LAPACKE_dpoequ( int matrix_order, lapack_int n, const double* a,
                           lapack_int lda, double* s, double* scond,
                           double* amax );
lapack_int LAPACKE_cpoequ( int matrix_order, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           float* s, float* scond, float* amax );
lapack_int LAPACKE_zpoequ( int matrix_order, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           double* s, double* scond, double* amax );

lapack_int LAPACKE_spoequb( int matrix_order, lapack_int n, const float* a,
                            lapack_int lda, float* s, float* scond,
                            float* amax );
lapack_int LAPACKE_dpoequb( int matrix_order, lapack_int n, const double* a,
                            lapack_int lda, double* s, double* scond,
                            double* amax );
lapack_int LAPACKE_cpoequb( int matrix_order, lapack_int n,
                            const lapack_complex_float* a, lapack_int lda,
                            float* s, float* scond, float* amax );
lapack_int LAPACKE_zpoequb( int matrix_order, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            double* s, double* scond, double* amax );

lapack_int LAPACKE_sporfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const float* af, lapack_int ldaf, const float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dporfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_cporfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* af,
                           lapack_int ldaf, const lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_zporfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* af,
                           lapack_int ldaf, const lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, double* ferr, double* berr );

lapack_int LAPACKE_sporfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs, const float* a,
                            lapack_int lda, const float* af, lapack_int ldaf,
                            const float* s, const float* b, lapack_int ldb,
                            float* x, lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dporfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs, const double* a,
                            lapack_int lda, const double* af, lapack_int ldaf,
                            const double* s, const double* b, lapack_int ldb,
                            double* x, lapack_int ldx, double* rcond,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );
lapack_int LAPACKE_cporfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_float* a, lapack_int lda,
                            const lapack_complex_float* af, lapack_int ldaf,
                            const float* s, const lapack_complex_float* b,
                            lapack_int ldb, lapack_complex_float* x,
                            lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_zporfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_double* a, lapack_int lda,
                            const lapack_complex_double* af, lapack_int ldaf,
                            const double* s, const lapack_complex_double* b,
                            lapack_int ldb, lapack_complex_double* x,
                            lapack_int ldx, double* rcond, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );

lapack_int LAPACKE_sposv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, float* a, lapack_int lda, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dposv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cposv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zposv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_complex_double* b,
                          lapack_int ldb );
lapack_int LAPACKE_dsposv( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           lapack_int* iter );
lapack_int LAPACKE_zcposv( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* x,
                           lapack_int ldx, lapack_int* iter );

lapack_int LAPACKE_sposvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, float* a, lapack_int lda, float* af,
                           lapack_int ldaf, char* equed, float* s, float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_dposvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, double* a, lapack_int lda,
                           double* af, lapack_int ldaf, char* equed, double* s,
                           double* b, lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
lapack_int LAPACKE_cposvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* af,
                           lapack_int ldaf, char* equed, float* s,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zposvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* af,
                           lapack_int ldaf, char* equed, double* s,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_sposvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs, float* a,
                            lapack_int lda, float* af, lapack_int ldaf,
                            char* equed, float* s, float* b, lapack_int ldb,
                            float* x, lapack_int ldx, float* rcond,
                            float* rpvgrw, float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_dposvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs, double* a,
                            lapack_int lda, double* af, lapack_int ldaf,
                            char* equed, double* s, double* b, lapack_int ldb,
                            double* x, lapack_int ldx, double* rcond,
                            double* rpvgrw, double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );
lapack_int LAPACKE_cposvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* af, lapack_int ldaf,
                            char* equed, float* s, lapack_complex_float* b,
                            lapack_int ldb, lapack_complex_float* x,
                            lapack_int ldx, float* rcond, float* rpvgrw,
                            float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_zposvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* af, lapack_int ldaf,
                            char* equed, double* s, lapack_complex_double* b,
                            lapack_int ldb, lapack_complex_double* x,
                            lapack_int ldx, double* rcond, double* rpvgrw,
                            double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );

lapack_int LAPACKE_spotrf( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dpotrf( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_cpotrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_spotri( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dpotri( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda );
lapack_int LAPACKE_cpotri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_zpotri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_spotrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dpotrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_cpotrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zpotrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_sppcon( int matrix_order, char uplo, lapack_int n,
                           const float* ap, float anorm, float* rcond );
lapack_int LAPACKE_dppcon( int matrix_order, char uplo, lapack_int n,
                           const double* ap, double anorm, double* rcond );
lapack_int LAPACKE_cppcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap, float anorm,
                           float* rcond );
lapack_int LAPACKE_zppcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap, double anorm,
                           double* rcond );

lapack_int LAPACKE_sppequ( int matrix_order, char uplo, lapack_int n,
                           const float* ap, float* s, float* scond,
                           float* amax );
lapack_int LAPACKE_dppequ( int matrix_order, char uplo, lapack_int n,
                           const double* ap, double* s, double* scond,
                           double* amax );
lapack_int LAPACKE_cppequ( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap, float* s,
                           float* scond, float* amax );
lapack_int LAPACKE_zppequ( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap, double* s,
                           double* scond, double* amax );

lapack_int LAPACKE_spprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap, const float* afp,
                           const float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dpprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, const double* afp,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_cpprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_complex_float* afp,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zpprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_complex_double* afp,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sppsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, float* ap, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dppsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, double* ap, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_cppsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zppsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_sppsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, float* ap, float* afp, char* equed,
                           float* s, float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dppsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, double* ap, double* afp,
                           char* equed, double* s, double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
lapack_int LAPACKE_cppsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_float* ap,
                           lapack_complex_float* afp, char* equed, float* s,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zppsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, lapack_complex_double* ap,
                           lapack_complex_double* afp, char* equed, double* s,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_spptrf( int matrix_order, char uplo, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dpptrf( int matrix_order, char uplo, lapack_int n,
                           double* ap );
lapack_int LAPACKE_cpptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_zpptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap );

lapack_int LAPACKE_spptri( int matrix_order, char uplo, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dpptri( int matrix_order, char uplo, lapack_int n,
                           double* ap );
lapack_int LAPACKE_cpptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_zpptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap );

lapack_int LAPACKE_spptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dpptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_cpptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_spstrf( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda, lapack_int* piv, lapack_int* rank,
                           float tol );
lapack_int LAPACKE_dpstrf( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda, lapack_int* piv, lapack_int* rank,
                           double tol );
lapack_int LAPACKE_cpstrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* piv, lapack_int* rank, float tol );
lapack_int LAPACKE_zpstrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* piv, lapack_int* rank, double tol );

lapack_int LAPACKE_sptcon( lapack_int n, const float* d, const float* e,
                           float anorm, float* rcond );
lapack_int LAPACKE_dptcon( lapack_int n, const double* d, const double* e,
                           double anorm, double* rcond );
lapack_int LAPACKE_cptcon( lapack_int n, const float* d,
                           const lapack_complex_float* e, float anorm,
                           float* rcond );
lapack_int LAPACKE_zptcon( lapack_int n, const double* d,
                           const lapack_complex_double* e, double anorm,
                           double* rcond );

lapack_int LAPACKE_spteqr( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dpteqr( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_cpteqr( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zpteqr( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_sptrfs( int matrix_order, lapack_int n, lapack_int nrhs,
                           const float* d, const float* e, const float* df,
                           const float* ef, const float* b, lapack_int ldb,
                           float* x, lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dptrfs( int matrix_order, lapack_int n, lapack_int nrhs,
                           const double* d, const double* e, const double* df,
                           const double* ef, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* ferr,
                           double* berr );
lapack_int LAPACKE_cptrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* d,
                           const lapack_complex_float* e, const float* df,
                           const lapack_complex_float* ef,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zptrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* d,
                           const lapack_complex_double* e, const double* df,
                           const lapack_complex_double* ef,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sptsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          float* d, float* e, float* b, lapack_int ldb );
lapack_int LAPACKE_dptsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          double* d, double* e, double* b, lapack_int ldb );
lapack_int LAPACKE_cptsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          float* d, lapack_complex_float* e,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zptsv( int matrix_order, lapack_int n, lapack_int nrhs,
                          double* d, lapack_complex_double* e,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_sptsvx( int matrix_order, char fact, lapack_int n,
                           lapack_int nrhs, const float* d, const float* e,
                           float* df, float* ef, const float* b, lapack_int ldb,
                           float* x, lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dptsvx( int matrix_order, char fact, lapack_int n,
                           lapack_int nrhs, const double* d, const double* e,
                           double* df, double* ef, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );
lapack_int LAPACKE_cptsvx( int matrix_order, char fact, lapack_int n,
                           lapack_int nrhs, const float* d,
                           const lapack_complex_float* e, float* df,
                           lapack_complex_float* ef,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zptsvx( int matrix_order, char fact, lapack_int n,
                           lapack_int nrhs, const double* d,
                           const lapack_complex_double* e, double* df,
                           lapack_complex_double* ef,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_spttrf( lapack_int n, float* d, float* e );
lapack_int LAPACKE_dpttrf( lapack_int n, double* d, double* e );
lapack_int LAPACKE_cpttrf( lapack_int n, float* d, lapack_complex_float* e );
lapack_int LAPACKE_zpttrf( lapack_int n, double* d, lapack_complex_double* e );

lapack_int LAPACKE_spttrs( int matrix_order, lapack_int n, lapack_int nrhs,
                           const float* d, const float* e, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dpttrs( int matrix_order, lapack_int n, lapack_int nrhs,
                           const double* d, const double* e, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_cpttrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* d,
                           const lapack_complex_float* e,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zpttrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* d,
                           const lapack_complex_double* e,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_ssbev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int kd, float* ab, lapack_int ldab, float* w,
                          float* z, lapack_int ldz );
lapack_int LAPACKE_dsbev( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int kd, double* ab, lapack_int ldab, double* w,
                          double* z, lapack_int ldz );

lapack_int LAPACKE_ssbevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int kd, float* ab, lapack_int ldab, float* w,
                           float* z, lapack_int ldz );
lapack_int LAPACKE_dsbevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int kd, double* ab, lapack_int ldab,
                           double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_ssbevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd, float* ab,
                           lapack_int ldab, float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsbevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int kd, double* ab,
                           lapack_int ldab, double* q, lapack_int ldq,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );

lapack_int LAPACKE_ssbgst( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, float* ab,
                           lapack_int ldab, const float* bb, lapack_int ldbb,
                           float* x, lapack_int ldx );
lapack_int LAPACKE_dsbgst( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, double* ab,
                           lapack_int ldab, const double* bb, lapack_int ldbb,
                           double* x, lapack_int ldx );

lapack_int LAPACKE_ssbgv( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb, float* ab,
                          lapack_int ldab, float* bb, lapack_int ldbb, float* w,
                          float* z, lapack_int ldz );
lapack_int LAPACKE_dsbgv( int matrix_order, char jobz, char uplo, lapack_int n,
                          lapack_int ka, lapack_int kb, double* ab,
                          lapack_int ldab, double* bb, lapack_int ldbb,
                          double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_ssbgvd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, float* ab,
                           lapack_int ldab, float* bb, lapack_int ldbb,
                           float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dsbgvd( int matrix_order, char jobz, char uplo, lapack_int n,
                           lapack_int ka, lapack_int kb, double* ab,
                           lapack_int ldab, double* bb, lapack_int ldbb,
                           double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_ssbgvx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           float* ab, lapack_int ldab, float* bb,
                           lapack_int ldbb, float* q, lapack_int ldq, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsbgvx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, lapack_int ka, lapack_int kb,
                           double* ab, lapack_int ldab, double* bb,
                           lapack_int ldbb, double* q, lapack_int ldq,
                           double vl, double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );

lapack_int LAPACKE_ssbtrd( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int kd, float* ab, lapack_int ldab, float* d,
                           float* e, float* q, lapack_int ldq );
lapack_int LAPACKE_dsbtrd( int matrix_order, char vect, char uplo, lapack_int n,
                           lapack_int kd, double* ab, lapack_int ldab,
                           double* d, double* e, double* q, lapack_int ldq );

lapack_int LAPACKE_ssfrk( int matrix_order, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, float alpha,
                          const float* a, lapack_int lda, float beta,
                          float* c );
lapack_int LAPACKE_dsfrk( int matrix_order, char transr, char uplo, char trans,
                          lapack_int n, lapack_int k, double alpha,
                          const double* a, lapack_int lda, double beta,
                          double* c );

lapack_int LAPACKE_sspcon( int matrix_order, char uplo, lapack_int n,
                           const float* ap, const lapack_int* ipiv, float anorm,
                           float* rcond );
lapack_int LAPACKE_dspcon( int matrix_order, char uplo, lapack_int n,
                           const double* ap, const lapack_int* ipiv,
                           double anorm, double* rcond );
lapack_int LAPACKE_cspcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zspcon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_sspev( int matrix_order, char jobz, char uplo, lapack_int n,
                          float* ap, float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspev( int matrix_order, char jobz, char uplo, lapack_int n,
                          double* ap, double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_sspevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           float* ap, float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           double* ap, double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_sspevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, float* ap, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dspevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, double* ap, double vl, double vu,
                           lapack_int il, lapack_int iu, double abstol,
                           lapack_int* m, double* w, double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_sspgst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, float* ap, const float* bp );
lapack_int LAPACKE_dspgst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, double* ap, const double* bp );

lapack_int LAPACKE_sspgv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, float* ap, float* bp,
                          float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspgv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, double* ap, double* bp,
                          double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_sspgvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, float* ap, float* bp,
                           float* w, float* z, lapack_int ldz );
lapack_int LAPACKE_dspgvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, double* ap, double* bp,
                           double* w, double* z, lapack_int ldz );

lapack_int LAPACKE_sspgvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, float* ap,
                           float* bp, float vl, float vu, lapack_int il,
                           lapack_int iu, float abstol, lapack_int* m, float* w,
                           float* z, lapack_int ldz, lapack_int* ifail );
lapack_int LAPACKE_dspgvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, double* ap,
                           double* bp, double vl, double vu, lapack_int il,
                           lapack_int iu, double abstol, lapack_int* m,
                           double* w, double* z, lapack_int ldz,
                           lapack_int* ifail );

lapack_int LAPACKE_ssprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap, const float* afp,
                           const lapack_int* ipiv, const float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dsprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, const double* afp,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_csprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_complex_float* afp,
                           const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zsprfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_complex_double* afp,
                           const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_sspsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, float* ap, lapack_int* ipiv,
                          float* b, lapack_int ldb );
lapack_int LAPACKE_dspsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, double* ap, lapack_int* ipiv,
                          double* b, lapack_int ldb );
lapack_int LAPACKE_cspsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* ap,
                          lapack_int* ipiv, lapack_complex_float* b,
                          lapack_int ldb );
lapack_int LAPACKE_zspsv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* ap,
                          lapack_int* ipiv, lapack_complex_double* b,
                          lapack_int ldb );

lapack_int LAPACKE_sspsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap, float* afp,
                           lapack_int* ipiv, const float* b, lapack_int ldb,
                           float* x, lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dspsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap, double* afp,
                           lapack_int* ipiv, const double* b, lapack_int ldb,
                           double* x, lapack_int ldx, double* rcond,
                           double* ferr, double* berr );
lapack_int LAPACKE_cspsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           lapack_complex_float* afp, lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zspsvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           lapack_complex_double* afp, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_ssptrd( int matrix_order, char uplo, lapack_int n, float* ap,
                           float* d, float* e, float* tau );
lapack_int LAPACKE_dsptrd( int matrix_order, char uplo, lapack_int n,
                           double* ap, double* d, double* e, double* tau );

lapack_int LAPACKE_ssptrf( int matrix_order, char uplo, lapack_int n, float* ap,
                           lapack_int* ipiv );
lapack_int LAPACKE_dsptrf( int matrix_order, char uplo, lapack_int n,
                           double* ap, lapack_int* ipiv );
lapack_int LAPACKE_csptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap, lapack_int* ipiv );
lapack_int LAPACKE_zsptrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap, lapack_int* ipiv );

lapack_int LAPACKE_ssptri( int matrix_order, char uplo, lapack_int n, float* ap,
                           const lapack_int* ipiv );
lapack_int LAPACKE_dsptri( int matrix_order, char uplo, lapack_int n,
                           double* ap, const lapack_int* ipiv );
lapack_int LAPACKE_csptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* ap, const lapack_int* ipiv );
lapack_int LAPACKE_zsptri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* ap, const lapack_int* ipiv );

lapack_int LAPACKE_ssptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* ap,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* ap,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* ap,
                           const lapack_int* ipiv, lapack_complex_float* b,
                           lapack_int ldb );
lapack_int LAPACKE_zsptrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* ap,
                           const lapack_int* ipiv, lapack_complex_double* b,
                           lapack_int ldb );

lapack_int LAPACKE_sstebz( char range, char order, lapack_int n, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           const float* d, const float* e, lapack_int* m,
                           lapack_int* nsplit, float* w, lapack_int* iblock,
                           lapack_int* isplit );
lapack_int LAPACKE_dstebz( char range, char order, lapack_int n, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, const double* d, const double* e,
                           lapack_int* m, lapack_int* nsplit, double* w,
                           lapack_int* iblock, lapack_int* isplit );

lapack_int LAPACKE_sstedc( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dstedc( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_cstedc( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zstedc( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_sstegr( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_dstegr( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* isuppz );
lapack_int LAPACKE_cstegr( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* isuppz );
lapack_int LAPACKE_zstegr( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* isuppz );

lapack_int LAPACKE_sstein( int matrix_order, lapack_int n, const float* d,
                           const float* e, lapack_int m, const float* w,
                           const lapack_int* iblock, const lapack_int* isplit,
                           float* z, lapack_int ldz, lapack_int* ifailv );
lapack_int LAPACKE_dstein( int matrix_order, lapack_int n, const double* d,
                           const double* e, lapack_int m, const double* w,
                           const lapack_int* iblock, const lapack_int* isplit,
                           double* z, lapack_int ldz, lapack_int* ifailv );
lapack_int LAPACKE_cstein( int matrix_order, lapack_int n, const float* d,
                           const float* e, lapack_int m, const float* w,
                           const lapack_int* iblock, const lapack_int* isplit,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int* ifailv );
lapack_int LAPACKE_zstein( int matrix_order, lapack_int n, const double* d,
                           const double* e, lapack_int m, const double* w,
                           const lapack_int* iblock, const lapack_int* isplit,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* ifailv );

lapack_int LAPACKE_sstemr( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, lapack_int* m,
                           float* w, float* z, lapack_int ldz, lapack_int nzc,
                           lapack_int* isuppz, lapack_logical* tryrac );
lapack_int LAPACKE_dstemr( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           lapack_int* m, double* w, double* z, lapack_int ldz,
                           lapack_int nzc, lapack_int* isuppz,
                           lapack_logical* tryrac );
lapack_int LAPACKE_cstemr( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, lapack_int* m,
                           float* w, lapack_complex_float* z, lapack_int ldz,
                           lapack_int nzc, lapack_int* isuppz,
                           lapack_logical* tryrac );
lapack_int LAPACKE_zstemr( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           lapack_int* m, double* w, lapack_complex_double* z,
                           lapack_int ldz, lapack_int nzc, lapack_int* isuppz,
                           lapack_logical* tryrac );

lapack_int LAPACKE_ssteqr( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dsteqr( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, double* z, lapack_int ldz );
lapack_int LAPACKE_csteqr( int matrix_order, char compz, lapack_int n, float* d,
                           float* e, lapack_complex_float* z, lapack_int ldz );
lapack_int LAPACKE_zsteqr( int matrix_order, char compz, lapack_int n,
                           double* d, double* e, lapack_complex_double* z,
                           lapack_int ldz );

lapack_int LAPACKE_ssterf( lapack_int n, float* d, float* e );
lapack_int LAPACKE_dsterf( lapack_int n, double* d, double* e );

lapack_int LAPACKE_sstev( int matrix_order, char jobz, lapack_int n, float* d,
                          float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dstev( int matrix_order, char jobz, lapack_int n, double* d,
                          double* e, double* z, lapack_int ldz );

lapack_int LAPACKE_sstevd( int matrix_order, char jobz, lapack_int n, float* d,
                           float* e, float* z, lapack_int ldz );
lapack_int LAPACKE_dstevd( int matrix_order, char jobz, lapack_int n, double* d,
                           double* e, double* z, lapack_int ldz );

lapack_int LAPACKE_sstevr( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_dstevr( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* isuppz );

lapack_int LAPACKE_sstevx( int matrix_order, char jobz, char range,
                           lapack_int n, float* d, float* e, float vl, float vu,
                           lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dstevx( int matrix_order, char jobz, char range,
                           lapack_int n, double* d, double* e, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );

lapack_int LAPACKE_ssycon( int matrix_order, char uplo, lapack_int n,
                           const float* a, lapack_int lda,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_dsycon( int matrix_order, char uplo, lapack_int n,
                           const double* a, lapack_int lda,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );
lapack_int LAPACKE_csycon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv, float anorm, float* rcond );
lapack_int LAPACKE_zsycon( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv, double anorm,
                           double* rcond );

lapack_int LAPACKE_ssyconv( int matrix_order, char uplo, char way, lapack_int n,
                            float* a, lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dsyconv( int matrix_order, char uplo, char way, lapack_int n,
                            double* a, lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_csyconv( int matrix_order, char uplo, char way, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            const lapack_int* ipiv );
lapack_int LAPACKE_zsyconv( int matrix_order, char uplo, char way, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            const lapack_int* ipiv );

lapack_int LAPACKE_ssyequb( int matrix_order, char uplo, lapack_int n,
                            const float* a, lapack_int lda, float* s,
                            float* scond, float* amax );
lapack_int LAPACKE_dsyequb( int matrix_order, char uplo, lapack_int n,
                            const double* a, lapack_int lda, double* s,
                            double* scond, double* amax );
lapack_int LAPACKE_csyequb( int matrix_order, char uplo, lapack_int n,
                            const lapack_complex_float* a, lapack_int lda,
                            float* s, float* scond, float* amax );
lapack_int LAPACKE_zsyequb( int matrix_order, char uplo, lapack_int n,
                            const lapack_complex_double* a, lapack_int lda,
                            double* s, double* scond, double* amax );

lapack_int LAPACKE_ssyev( int matrix_order, char jobz, char uplo, lapack_int n,
                          float* a, lapack_int lda, float* w );
lapack_int LAPACKE_dsyev( int matrix_order, char jobz, char uplo, lapack_int n,
                          double* a, lapack_int lda, double* w );

lapack_int LAPACKE_ssyevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           float* a, lapack_int lda, float* w );
lapack_int LAPACKE_dsyevd( int matrix_order, char jobz, char uplo, lapack_int n,
                           double* a, lapack_int lda, double* w );

lapack_int LAPACKE_ssyevr( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, float* a, lapack_int lda, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* isuppz );
lapack_int LAPACKE_dsyevr( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, double* a, lapack_int lda, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* isuppz );

lapack_int LAPACKE_ssyevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, float* a, lapack_int lda, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsyevx( int matrix_order, char jobz, char range, char uplo,
                           lapack_int n, double* a, lapack_int lda, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );

lapack_int LAPACKE_ssygst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, float* a, lapack_int lda,
                           const float* b, lapack_int ldb );
lapack_int LAPACKE_dsygst( int matrix_order, lapack_int itype, char uplo,
                           lapack_int n, double* a, lapack_int lda,
                           const double* b, lapack_int ldb );

lapack_int LAPACKE_ssygv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, float* a, lapack_int lda,
                          float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_dsygv( int matrix_order, lapack_int itype, char jobz,
                          char uplo, lapack_int n, double* a, lapack_int lda,
                          double* b, lapack_int ldb, double* w );

lapack_int LAPACKE_ssygvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, float* a, lapack_int lda,
                           float* b, lapack_int ldb, float* w );
lapack_int LAPACKE_dsygvd( int matrix_order, lapack_int itype, char jobz,
                           char uplo, lapack_int n, double* a, lapack_int lda,
                           double* b, lapack_int ldb, double* w );

lapack_int LAPACKE_ssygvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float vl,
                           float vu, lapack_int il, lapack_int iu, float abstol,
                           lapack_int* m, float* w, float* z, lapack_int ldz,
                           lapack_int* ifail );
lapack_int LAPACKE_dsygvx( int matrix_order, lapack_int itype, char jobz,
                           char range, char uplo, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb, double vl,
                           double vu, lapack_int il, lapack_int iu,
                           double abstol, lapack_int* m, double* w, double* z,
                           lapack_int ldz, lapack_int* ifail );

lapack_int LAPACKE_ssyrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const float* af, lapack_int ldaf,
                           const lapack_int* ipiv, const float* b,
                           lapack_int ldb, float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dsyrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const double* af, lapack_int ldaf,
                           const lapack_int* ipiv, const double* b,
                           lapack_int ldb, double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_csyrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_zsyrfs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* af,
                           lapack_int ldaf, const lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_ssyrfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs, const float* a,
                            lapack_int lda, const float* af, lapack_int ldaf,
                            const lapack_int* ipiv, const float* s,
                            const float* b, lapack_int ldb, float* x,
                            lapack_int ldx, float* rcond, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dsyrfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs, const double* a,
                            lapack_int lda, const double* af, lapack_int ldaf,
                            const lapack_int* ipiv, const double* s,
                            const double* b, lapack_int ldb, double* x,
                            lapack_int ldx, double* rcond, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );
lapack_int LAPACKE_csyrfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_float* a, lapack_int lda,
                            const lapack_complex_float* af, lapack_int ldaf,
                            const lapack_int* ipiv, const float* s,
                            const lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* berr, lapack_int n_err_bnds,
                            float* err_bnds_norm, float* err_bnds_comp,
                            lapack_int nparams, float* params );
lapack_int LAPACKE_zsyrfsx( int matrix_order, char uplo, char equed,
                            lapack_int n, lapack_int nrhs,
                            const lapack_complex_double* a, lapack_int lda,
                            const lapack_complex_double* af, lapack_int ldaf,
                            const lapack_int* ipiv, const double* s,
                            const lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* berr, lapack_int n_err_bnds,
                            double* err_bnds_norm, double* err_bnds_comp,
                            lapack_int nparams, double* params );

lapack_int LAPACKE_ssysv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, float* a, lapack_int lda,
                          lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsysv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, double* a, lapack_int lda,
                          lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csysv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_float* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsysv( int matrix_order, char uplo, lapack_int n,
                          lapack_int nrhs, lapack_complex_double* a,
                          lapack_int lda, lapack_int* ipiv,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_ssysvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           float* af, lapack_int ldaf, lapack_int* ipiv,
                           const float* b, lapack_int ldb, float* x,
                           lapack_int ldx, float* rcond, float* ferr,
                           float* berr );
lapack_int LAPACKE_dsysvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           double* af, lapack_int ldaf, lapack_int* ipiv,
                           const double* b, lapack_int ldb, double* x,
                           lapack_int ldx, double* rcond, double* ferr,
                           double* berr );
lapack_int LAPACKE_csysvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* af,
                           lapack_int ldaf, lapack_int* ipiv,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* x, lapack_int ldx,
                           float* rcond, float* ferr, float* berr );
lapack_int LAPACKE_zsysvx( int matrix_order, char fact, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* af,
                           lapack_int ldaf, lapack_int* ipiv,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* x, lapack_int ldx,
                           double* rcond, double* ferr, double* berr );

lapack_int LAPACKE_ssysvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs, float* a,
                            lapack_int lda, float* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, float* s, float* b,
                            lapack_int ldb, float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_dsysvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs, double* a,
                            lapack_int lda, double* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, double* s, double* b,
                            lapack_int ldb, double* x, lapack_int ldx,
                            double* rcond, double* rpvgrw, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );
lapack_int LAPACKE_csysvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, float* s,
                            lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* x, lapack_int ldx,
                            float* rcond, float* rpvgrw, float* berr,
                            lapack_int n_err_bnds, float* err_bnds_norm,
                            float* err_bnds_comp, lapack_int nparams,
                            float* params );
lapack_int LAPACKE_zsysvxx( int matrix_order, char fact, char uplo,
                            lapack_int n, lapack_int nrhs,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* af, lapack_int ldaf,
                            lapack_int* ipiv, char* equed, double* s,
                            lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* x, lapack_int ldx,
                            double* rcond, double* rpvgrw, double* berr,
                            lapack_int n_err_bnds, double* err_bnds_norm,
                            double* err_bnds_comp, lapack_int nparams,
                            double* params );

lapack_int LAPACKE_ssyswapr( int matrix_order, char uplo, lapack_int n,
                             float* a, lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_dsyswapr( int matrix_order, char uplo, lapack_int n,
                             double* a, lapack_int i1, lapack_int i2 );
lapack_int LAPACKE_csyswapr( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int i1,
                             lapack_int i2 );
lapack_int LAPACKE_zsyswapr( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int i1,
                             lapack_int i2 );

lapack_int LAPACKE_ssytrd( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda, float* d, float* e, float* tau );
lapack_int LAPACKE_dsytrd( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda, double* d, double* e, double* tau );

lapack_int LAPACKE_ssytrf( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_dsytrf( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda, lapack_int* ipiv );
lapack_int LAPACKE_csytrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_int* ipiv );
lapack_int LAPACKE_zsytrf( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_int* ipiv );

lapack_int LAPACKE_ssytri( int matrix_order, char uplo, lapack_int n, float* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dsytri( int matrix_order, char uplo, lapack_int n, double* a,
                           lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_csytri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_int* ipiv );
lapack_int LAPACKE_zsytri( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_int* ipiv );

lapack_int LAPACKE_ssytri2( int matrix_order, char uplo, lapack_int n, float* a,
                            lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_dsytri2( int matrix_order, char uplo, lapack_int n,
                            double* a, lapack_int lda, const lapack_int* ipiv );
lapack_int LAPACKE_csytri2( int matrix_order, char uplo, lapack_int n,
                            lapack_complex_float* a, lapack_int lda,
                            const lapack_int* ipiv );
lapack_int LAPACKE_zsytri2( int matrix_order, char uplo, lapack_int n,
                            lapack_complex_double* a, lapack_int lda,
                            const lapack_int* ipiv );

lapack_int LAPACKE_ssytri2x( int matrix_order, char uplo, lapack_int n,
                             float* a, lapack_int lda, const lapack_int* ipiv,
                             lapack_int nb );
lapack_int LAPACKE_dsytri2x( int matrix_order, char uplo, lapack_int n,
                             double* a, lapack_int lda, const lapack_int* ipiv,
                             lapack_int nb );
lapack_int LAPACKE_csytri2x( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_float* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );
lapack_int LAPACKE_zsytri2x( int matrix_order, char uplo, lapack_int n,
                             lapack_complex_double* a, lapack_int lda,
                             const lapack_int* ipiv, lapack_int nb );

lapack_int LAPACKE_ssytrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const float* a, lapack_int lda,
                           const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsytrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const double* a, lapack_int lda,
                           const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csytrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_float* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsytrs( int matrix_order, char uplo, lapack_int n,
                           lapack_int nrhs, const lapack_complex_double* a,
                           lapack_int lda, const lapack_int* ipiv,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_ssytrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const float* a, lapack_int lda,
                            const lapack_int* ipiv, float* b, lapack_int ldb );
lapack_int LAPACKE_dsytrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const double* a, lapack_int lda,
                            const lapack_int* ipiv, double* b, lapack_int ldb );
lapack_int LAPACKE_csytrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_float* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_zsytrs2( int matrix_order, char uplo, lapack_int n,
                            lapack_int nrhs, const lapack_complex_double* a,
                            lapack_int lda, const lapack_int* ipiv,
                            lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_stbcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd, const float* ab,
                           lapack_int ldab, float* rcond );
lapack_int LAPACKE_dtbcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd, const double* ab,
                           lapack_int ldab, double* rcond );
lapack_int LAPACKE_ctbcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd,
                           const lapack_complex_float* ab, lapack_int ldab,
                           float* rcond );
lapack_int LAPACKE_ztbcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, lapack_int kd,
                           const lapack_complex_double* ab, lapack_int ldab,
                           double* rcond );

lapack_int LAPACKE_stbrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const float* ab, lapack_int ldab, const float* b,
                           lapack_int ldb, const float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_dtbrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const double* ab, lapack_int ldab, const double* b,
                           lapack_int ldb, const double* x, lapack_int ldx,
                           double* ferr, double* berr );
lapack_int LAPACKE_ctbrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztbrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_stbtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const float* ab, lapack_int ldab, float* b,
                           lapack_int ldb );
lapack_int LAPACKE_dtbtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const double* ab, lapack_int ldab, double* b,
                           lapack_int ldb );
lapack_int LAPACKE_ctbtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_float* ab, lapack_int ldab,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztbtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int kd, lapack_int nrhs,
                           const lapack_complex_double* ab, lapack_int ldab,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_stfsm( int matrix_order, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          float alpha, const float* a, float* b,
                          lapack_int ldb );
lapack_int LAPACKE_dtfsm( int matrix_order, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          double alpha, const double* a, double* b,
                          lapack_int ldb );
lapack_int LAPACKE_ctfsm( int matrix_order, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          lapack_complex_float alpha,
                          const lapack_complex_float* a,
                          lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztfsm( int matrix_order, char transr, char side, char uplo,
                          char trans, char diag, lapack_int m, lapack_int n,
                          lapack_complex_double alpha,
                          const lapack_complex_double* a,
                          lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_stftri( int matrix_order, char transr, char uplo, char diag,
                           lapack_int n, float* a );
lapack_int LAPACKE_dtftri( int matrix_order, char transr, char uplo, char diag,
                           lapack_int n, double* a );
lapack_int LAPACKE_ctftri( int matrix_order, char transr, char uplo, char diag,
                           lapack_int n, lapack_complex_float* a );
lapack_int LAPACKE_ztftri( int matrix_order, char transr, char uplo, char diag,
                           lapack_int n, lapack_complex_double* a );

lapack_int LAPACKE_stfttp( int matrix_order, char transr, char uplo,
                           lapack_int n, const float* arf, float* ap );
lapack_int LAPACKE_dtfttp( int matrix_order, char transr, char uplo,
                           lapack_int n, const double* arf, double* ap );
lapack_int LAPACKE_ctfttp( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* arf,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztfttp( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* arf,
                           lapack_complex_double* ap );

lapack_int LAPACKE_stfttr( int matrix_order, char transr, char uplo,
                           lapack_int n, const float* arf, float* a,
                           lapack_int lda );
lapack_int LAPACKE_dtfttr( int matrix_order, char transr, char uplo,
                           lapack_int n, const double* arf, double* a,
                           lapack_int lda );
lapack_int LAPACKE_ctfttr( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* arf,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztfttr( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* arf,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_stgevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const float* s, lapack_int lds, const float* p,
                           lapack_int ldp, float* vl, lapack_int ldvl,
                           float* vr, lapack_int ldvr, lapack_int mm,
                           lapack_int* m );
lapack_int LAPACKE_dtgevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const double* s, lapack_int lds, const double* p,
                           lapack_int ldp, double* vl, lapack_int ldvl,
                           double* vr, lapack_int ldvr, lapack_int mm,
                           lapack_int* m );
lapack_int LAPACKE_ctgevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_float* s, lapack_int lds,
                           const lapack_complex_float* p, lapack_int ldp,
                           lapack_complex_float* vl, lapack_int ldvl,
                           lapack_complex_float* vr, lapack_int ldvr,
                           lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ztgevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_double* s, lapack_int lds,
                           const lapack_complex_double* p, lapack_int ldp,
                           lapack_complex_double* vl, lapack_int ldvl,
                           lapack_complex_double* vr, lapack_int ldvr,
                           lapack_int mm, lapack_int* m );

lapack_int LAPACKE_stgexc( int matrix_order, lapack_logical wantq,
                           lapack_logical wantz, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb, float* q,
                           lapack_int ldq, float* z, lapack_int ldz,
                           lapack_int* ifst, lapack_int* ilst );
lapack_int LAPACKE_dtgexc( int matrix_order, lapack_logical wantq,
                           lapack_logical wantz, lapack_int n, double* a,
                           lapack_int lda, double* b, lapack_int ldb, double* q,
                           lapack_int ldq, double* z, lapack_int ldz,
                           lapack_int* ifst, lapack_int* ilst );
lapack_int LAPACKE_ctgexc( int matrix_order, lapack_logical wantq,
                           lapack_logical wantz, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_complex_float* z, lapack_int ldz,
                           lapack_int ifst, lapack_int ilst );
lapack_int LAPACKE_ztgexc( int matrix_order, lapack_logical wantq,
                           lapack_logical wantz, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int ifst, lapack_int ilst );

lapack_int LAPACKE_stgsen( int matrix_order, lapack_int ijob,
                           lapack_logical wantq, lapack_logical wantz,
                           const lapack_logical* select, lapack_int n, float* a,
                           lapack_int lda, float* b, lapack_int ldb,
                           float* alphar, float* alphai, float* beta, float* q,
                           lapack_int ldq, float* z, lapack_int ldz,
                           lapack_int* m, float* pl, float* pr, float* dif );
lapack_int LAPACKE_dtgsen( int matrix_order, lapack_int ijob,
                           lapack_logical wantq, lapack_logical wantz,
                           const lapack_logical* select, lapack_int n,
                           double* a, lapack_int lda, double* b, lapack_int ldb,
                           double* alphar, double* alphai, double* beta,
                           double* q, lapack_int ldq, double* z, lapack_int ldz,
                           lapack_int* m, double* pl, double* pr, double* dif );
lapack_int LAPACKE_ctgsen( int matrix_order, lapack_int ijob,
                           lapack_logical wantq, lapack_logical wantz,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* alpha,
                           lapack_complex_float* beta, lapack_complex_float* q,
                           lapack_int ldq, lapack_complex_float* z,
                           lapack_int ldz, lapack_int* m, float* pl, float* pr,
                           float* dif );
lapack_int LAPACKE_ztgsen( int matrix_order, lapack_int ijob,
                           lapack_logical wantq, lapack_logical wantz,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* alpha,
                           lapack_complex_double* beta,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* z, lapack_int ldz,
                           lapack_int* m, double* pl, double* pr, double* dif );

lapack_int LAPACKE_stgsja( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_int k, lapack_int l, float* a, lapack_int lda,
                           float* b, lapack_int ldb, float tola, float tolb,
                           float* alpha, float* beta, float* u, lapack_int ldu,
                           float* v, lapack_int ldv, float* q, lapack_int ldq,
                           lapack_int* ncycle );
lapack_int LAPACKE_dtgsja( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_int k, lapack_int l, double* a,
                           lapack_int lda, double* b, lapack_int ldb,
                           double tola, double tolb, double* alpha,
                           double* beta, double* u, lapack_int ldu, double* v,
                           lapack_int ldv, double* q, lapack_int ldq,
                           lapack_int* ncycle );
lapack_int LAPACKE_ctgsja( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_int k, lapack_int l, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, float tola, float tolb, float* alpha,
                           float* beta, lapack_complex_float* u, lapack_int ldu,
                           lapack_complex_float* v, lapack_int ldv,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_int* ncycle );
lapack_int LAPACKE_ztgsja( int matrix_order, char jobu, char jobv, char jobq,
                           lapack_int m, lapack_int p, lapack_int n,
                           lapack_int k, lapack_int l, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, double tola, double tolb,
                           double* alpha, double* beta,
                           lapack_complex_double* u, lapack_int ldu,
                           lapack_complex_double* v, lapack_int ldv,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_int* ncycle );

lapack_int LAPACKE_stgsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const float* a, lapack_int lda, const float* b,
                           lapack_int ldb, const float* vl, lapack_int ldvl,
                           const float* vr, lapack_int ldvr, float* s,
                           float* dif, lapack_int mm, lapack_int* m );
lapack_int LAPACKE_dtgsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const double* a, lapack_int lda, const double* b,
                           lapack_int ldb, const double* vl, lapack_int ldvl,
                           const double* vr, lapack_int ldvr, double* s,
                           double* dif, lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ctgsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* vl, lapack_int ldvl,
                           const lapack_complex_float* vr, lapack_int ldvr,
                           float* s, float* dif, lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ztgsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* vl, lapack_int ldvl,
                           const lapack_complex_double* vr, lapack_int ldvr,
                           double* s, double* dif, lapack_int mm,
                           lapack_int* m );

lapack_int LAPACKE_stgsyl( int matrix_order, char trans, lapack_int ijob,
                           lapack_int m, lapack_int n, const float* a,
                           lapack_int lda, const float* b, lapack_int ldb,
                           float* c, lapack_int ldc, const float* d,
                           lapack_int ldd, const float* e, lapack_int lde,
                           float* f, lapack_int ldf, float* scale, float* dif );
lapack_int LAPACKE_dtgsyl( int matrix_order, char trans, lapack_int ijob,
                           lapack_int m, lapack_int n, const double* a,
                           lapack_int lda, const double* b, lapack_int ldb,
                           double* c, lapack_int ldc, const double* d,
                           lapack_int ldd, const double* e, lapack_int lde,
                           double* f, lapack_int ldf, double* scale,
                           double* dif );
lapack_int LAPACKE_ctgsyl( int matrix_order, char trans, lapack_int ijob,
                           lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* c, lapack_int ldc,
                           const lapack_complex_float* d, lapack_int ldd,
                           const lapack_complex_float* e, lapack_int lde,
                           lapack_complex_float* f, lapack_int ldf,
                           float* scale, float* dif );
lapack_int LAPACKE_ztgsyl( int matrix_order, char trans, lapack_int ijob,
                           lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* c, lapack_int ldc,
                           const lapack_complex_double* d, lapack_int ldd,
                           const lapack_complex_double* e, lapack_int lde,
                           lapack_complex_double* f, lapack_int ldf,
                           double* scale, double* dif );

lapack_int LAPACKE_stpcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const float* ap, float* rcond );
lapack_int LAPACKE_dtpcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const double* ap, double* rcond );
lapack_int LAPACKE_ctpcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_float* ap,
                           float* rcond );
lapack_int LAPACKE_ztpcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_double* ap,
                           double* rcond );

lapack_int LAPACKE_stpmqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k, lapack_int l,
                            lapack_int nb, const float* v, lapack_int ldv,
                            const float* t, lapack_int ldt, float* a, lapack_int lda,
                            float* b, lapack_int ldb );
lapack_int LAPACKE_dtpmqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k, lapack_int l,
                            lapack_int nb, const double* v, lapack_int ldv,
                            const double* t, lapack_int ldt, double* a, lapack_int lda,
                            double* b, lapack_int ldb );
lapack_int LAPACKE_ctpmqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k, lapack_int l,
                            lapack_int nb, const lapack_complex_float* v,
                            lapack_int ldv, const lapack_complex_float* t,
                            lapack_int ldt, lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztpmqrt( int matrix_order, char side, char trans,
                            lapack_int m, lapack_int n, lapack_int k, lapack_int l,
                            lapack_int nb, const lapack_complex_double* v,
                            lapack_int ldv, const lapack_complex_double* t,
                            lapack_int ldt, lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_stpqrt( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                           lapack_int nb, float* a, lapack_int lda, float* b, lapack_int ldb,
                           float* t, lapack_int ldt );
lapack_int LAPACKE_dtpqrt( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                           lapack_int nb, double* a, lapack_int lda, double* b, lapack_int ldb,
                           double* t, lapack_int ldt );
lapack_int LAPACKE_ctpqrt( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                           lapack_int nb, lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* b,
                           lapack_int ldb, lapack_complex_float* t,
                           lapack_int ldt );
lapack_int LAPACKE_ztpqrt( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                           lapack_int nb, lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* b,
                           lapack_int ldb, lapack_complex_double* t,
                           lapack_int ldt );

lapack_int LAPACKE_stpqrt2( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                            float* a, lapack_int lda, float* b, lapack_int ldb,
                            float* t, lapack_int ldt );
lapack_int LAPACKE_dtpqrt2( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                            double* a, lapack_int lda, double* b, lapack_int ldb,
                            double* t, lapack_int ldt );
lapack_int LAPACKE_ctpqrt2( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                            lapack_complex_float* a, lapack_int lda,
                            lapack_complex_float* b, lapack_int ldb,
                            lapack_complex_float* t, lapack_int ldt );
lapack_int LAPACKE_ztpqrt2( int matrix_order, lapack_int m, lapack_int n, lapack_int l,
                            lapack_complex_double* a, lapack_int lda,
                            lapack_complex_double* b, lapack_int ldb,
                            lapack_complex_double* t, lapack_int ldt );

lapack_int LAPACKE_stprfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* ap,
                           const float* b, lapack_int ldb, const float* x,
                           lapack_int ldx, float* ferr, float* berr );
lapack_int LAPACKE_dtprfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* ap,
                           const double* b, lapack_int ldb, const double* x,
                           lapack_int ldx, double* ferr, double* berr );
lapack_int LAPACKE_ctprfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztprfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_stptri( int matrix_order, char uplo, char diag, lapack_int n,
                           float* ap );
lapack_int LAPACKE_dtptri( int matrix_order, char uplo, char diag, lapack_int n,
                           double* ap );
lapack_int LAPACKE_ctptri( int matrix_order, char uplo, char diag, lapack_int n,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztptri( int matrix_order, char uplo, char diag, lapack_int n,
                           lapack_complex_double* ap );

lapack_int LAPACKE_stptrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* ap,
                           float* b, lapack_int ldb );
lapack_int LAPACKE_dtptrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* ap,
                           double* b, lapack_int ldb );
lapack_int LAPACKE_ctptrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* ap,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztptrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* ap,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_stpttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const float* ap, float* arf );
lapack_int LAPACKE_dtpttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const double* ap, double* arf );
lapack_int LAPACKE_ctpttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* ap,
                           lapack_complex_float* arf );
lapack_int LAPACKE_ztpttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* ap,
                           lapack_complex_double* arf );

lapack_int LAPACKE_stpttr( int matrix_order, char uplo, lapack_int n,
                           const float* ap, float* a, lapack_int lda );
lapack_int LAPACKE_dtpttr( int matrix_order, char uplo, lapack_int n,
                           const double* ap, double* a, lapack_int lda );
lapack_int LAPACKE_ctpttr( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztpttr( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_strcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const float* a, lapack_int lda,
                           float* rcond );
lapack_int LAPACKE_dtrcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const double* a, lapack_int lda,
                           double* rcond );
lapack_int LAPACKE_ctrcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_float* a,
                           lapack_int lda, float* rcond );
lapack_int LAPACKE_ztrcon( int matrix_order, char norm, char uplo, char diag,
                           lapack_int n, const lapack_complex_double* a,
                           lapack_int lda, double* rcond );

lapack_int LAPACKE_strevc( int matrix_order, char side, char howmny,
                           lapack_logical* select, lapack_int n, const float* t,
                           lapack_int ldt, float* vl, lapack_int ldvl,
                           float* vr, lapack_int ldvr, lapack_int mm,
                           lapack_int* m );
lapack_int LAPACKE_dtrevc( int matrix_order, char side, char howmny,
                           lapack_logical* select, lapack_int n,
                           const double* t, lapack_int ldt, double* vl,
                           lapack_int ldvl, double* vr, lapack_int ldvr,
                           lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ctrevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* vl, lapack_int ldvl,
                           lapack_complex_float* vr, lapack_int ldvr,
                           lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ztrevc( int matrix_order, char side, char howmny,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* vl, lapack_int ldvl,
                           lapack_complex_double* vr, lapack_int ldvr,
                           lapack_int mm, lapack_int* m );

lapack_int LAPACKE_strexc( int matrix_order, char compq, lapack_int n, float* t,
                           lapack_int ldt, float* q, lapack_int ldq,
                           lapack_int* ifst, lapack_int* ilst );
lapack_int LAPACKE_dtrexc( int matrix_order, char compq, lapack_int n,
                           double* t, lapack_int ldt, double* q, lapack_int ldq,
                           lapack_int* ifst, lapack_int* ilst );
lapack_int LAPACKE_ctrexc( int matrix_order, char compq, lapack_int n,
                           lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_int ifst, lapack_int ilst );
lapack_int LAPACKE_ztrexc( int matrix_order, char compq, lapack_int n,
                           lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_int ifst, lapack_int ilst );

lapack_int LAPACKE_strrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* a,
                           lapack_int lda, const float* b, lapack_int ldb,
                           const float* x, lapack_int ldx, float* ferr,
                           float* berr );
lapack_int LAPACKE_dtrrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* a,
                           lapack_int lda, const double* b, lapack_int ldb,
                           const double* x, lapack_int ldx, double* ferr,
                           double* berr );
lapack_int LAPACKE_ctrrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           const lapack_complex_float* x, lapack_int ldx,
                           float* ferr, float* berr );
lapack_int LAPACKE_ztrrfs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           const lapack_complex_double* x, lapack_int ldx,
                           double* ferr, double* berr );

lapack_int LAPACKE_strsen( int matrix_order, char job, char compq,
                           const lapack_logical* select, lapack_int n, float* t,
                           lapack_int ldt, float* q, lapack_int ldq, float* wr,
                           float* wi, lapack_int* m, float* s, float* sep );
lapack_int LAPACKE_dtrsen( int matrix_order, char job, char compq,
                           const lapack_logical* select, lapack_int n,
                           double* t, lapack_int ldt, double* q, lapack_int ldq,
                           double* wr, double* wi, lapack_int* m, double* s,
                           double* sep );
lapack_int LAPACKE_ctrsen( int matrix_order, char job, char compq,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_float* t, lapack_int ldt,
                           lapack_complex_float* q, lapack_int ldq,
                           lapack_complex_float* w, lapack_int* m, float* s,
                           float* sep );
lapack_int LAPACKE_ztrsen( int matrix_order, char job, char compq,
                           const lapack_logical* select, lapack_int n,
                           lapack_complex_double* t, lapack_int ldt,
                           lapack_complex_double* q, lapack_int ldq,
                           lapack_complex_double* w, lapack_int* m, double* s,
                           double* sep );

lapack_int LAPACKE_strsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const float* t, lapack_int ldt, const float* vl,
                           lapack_int ldvl, const float* vr, lapack_int ldvr,
                           float* s, float* sep, lapack_int mm, lapack_int* m );
lapack_int LAPACKE_dtrsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const double* t, lapack_int ldt, const double* vl,
                           lapack_int ldvl, const double* vr, lapack_int ldvr,
                           double* s, double* sep, lapack_int mm,
                           lapack_int* m );
lapack_int LAPACKE_ctrsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_float* t, lapack_int ldt,
                           const lapack_complex_float* vl, lapack_int ldvl,
                           const lapack_complex_float* vr, lapack_int ldvr,
                           float* s, float* sep, lapack_int mm, lapack_int* m );
lapack_int LAPACKE_ztrsna( int matrix_order, char job, char howmny,
                           const lapack_logical* select, lapack_int n,
                           const lapack_complex_double* t, lapack_int ldt,
                           const lapack_complex_double* vl, lapack_int ldvl,
                           const lapack_complex_double* vr, lapack_int ldvr,
                           double* s, double* sep, lapack_int mm,
                           lapack_int* m );

lapack_int LAPACKE_strsyl( int matrix_order, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const float* a, lapack_int lda, const float* b,
                           lapack_int ldb, float* c, lapack_int ldc,
                           float* scale );
lapack_int LAPACKE_dtrsyl( int matrix_order, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const double* a, lapack_int lda, const double* b,
                           lapack_int ldb, double* c, lapack_int ldc,
                           double* scale );
lapack_int LAPACKE_ctrsyl( int matrix_order, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* b, lapack_int ldb,
                           lapack_complex_float* c, lapack_int ldc,
                           float* scale );
lapack_int LAPACKE_ztrsyl( int matrix_order, char trana, char tranb,
                           lapack_int isgn, lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* b, lapack_int ldb,
                           lapack_complex_double* c, lapack_int ldc,
                           double* scale );

lapack_int LAPACKE_strtri( int matrix_order, char uplo, char diag, lapack_int n,
                           float* a, lapack_int lda );
lapack_int LAPACKE_dtrtri( int matrix_order, char uplo, char diag, lapack_int n,
                           double* a, lapack_int lda );
lapack_int LAPACKE_ctrtri( int matrix_order, char uplo, char diag, lapack_int n,
                           lapack_complex_float* a, lapack_int lda );
lapack_int LAPACKE_ztrtri( int matrix_order, char uplo, char diag, lapack_int n,
                           lapack_complex_double* a, lapack_int lda );

lapack_int LAPACKE_strtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const float* a,
                           lapack_int lda, float* b, lapack_int ldb );
lapack_int LAPACKE_dtrtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs, const double* a,
                           lapack_int lda, double* b, lapack_int ldb );
lapack_int LAPACKE_ctrtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* b, lapack_int ldb );
lapack_int LAPACKE_ztrtrs( int matrix_order, char uplo, char trans, char diag,
                           lapack_int n, lapack_int nrhs,
                           const lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* b, lapack_int ldb );

lapack_int LAPACKE_strttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const float* a, lapack_int lda,
                           float* arf );
lapack_int LAPACKE_dtrttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const double* a, lapack_int lda,
                           double* arf );
lapack_int LAPACKE_ctrttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_float* a,
                           lapack_int lda, lapack_complex_float* arf );
lapack_int LAPACKE_ztrttf( int matrix_order, char transr, char uplo,
                           lapack_int n, const lapack_complex_double* a,
                           lapack_int lda, lapack_complex_double* arf );

lapack_int LAPACKE_strttp( int matrix_order, char uplo, lapack_int n,
                           const float* a, lapack_int lda, float* ap );
lapack_int LAPACKE_dtrttp( int matrix_order, char uplo, lapack_int n,
                           const double* a, lapack_int lda, double* ap );
lapack_int LAPACKE_ctrttp( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* ap );
lapack_int LAPACKE_ztrttp( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* ap );

lapack_int LAPACKE_stzrzf( int matrix_order, lapack_int m, lapack_int n,
                           float* a, lapack_int lda, float* tau );
lapack_int LAPACKE_dtzrzf( int matrix_order, lapack_int m, lapack_int n,
                           double* a, lapack_int lda, double* tau );
lapack_int LAPACKE_ctzrzf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           lapack_complex_float* tau );
lapack_int LAPACKE_ztzrzf( int matrix_order, lapack_int m, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           lapack_complex_double* tau );

lapack_int LAPACKE_cunbdb( int matrix_order, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_float* x11, lapack_int ldx11,
                           lapack_complex_float* x12, lapack_int ldx12,
                           lapack_complex_float* x21, lapack_int ldx21,
                           lapack_complex_float* x22, lapack_int ldx22,
                           float* theta, float* phi,
                           lapack_complex_float* taup1,
                           lapack_complex_float* taup2,
                           lapack_complex_float* tauq1,
                           lapack_complex_float* tauq2 );
lapack_int LAPACKE_zunbdb( int matrix_order, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_double* x11, lapack_int ldx11,
                           lapack_complex_double* x12, lapack_int ldx12,
                           lapack_complex_double* x21, lapack_int ldx21,
                           lapack_complex_double* x22, lapack_int ldx22,
                           double* theta, double* phi,
                           lapack_complex_double* taup1,
                           lapack_complex_double* taup2,
                           lapack_complex_double* tauq1,
                           lapack_complex_double* tauq2 );

lapack_int LAPACKE_cuncsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_float* x11, lapack_int ldx11,
                           lapack_complex_float* x12, lapack_int ldx12,
                           lapack_complex_float* x21, lapack_int ldx21,
                           lapack_complex_float* x22, lapack_int ldx22,
                           float* theta, lapack_complex_float* u1,
                           lapack_int ldu1, lapack_complex_float* u2,
                           lapack_int ldu2, lapack_complex_float* v1t,
                           lapack_int ldv1t, lapack_complex_float* v2t,
                           lapack_int ldv2t );
lapack_int LAPACKE_zuncsd( int matrix_order, char jobu1, char jobu2,
                           char jobv1t, char jobv2t, char trans, char signs,
                           lapack_int m, lapack_int p, lapack_int q,
                           lapack_complex_double* x11, lapack_int ldx11,
                           lapack_complex_double* x12, lapack_int ldx12,
                           lapack_complex_double* x21, lapack_int ldx21,
                           lapack_complex_double* x22, lapack_int ldx22,
                           double* theta, lapack_complex_double* u1,
                           lapack_int ldu1, lapack_complex_double* u2,
                           lapack_int ldu2, lapack_complex_double* v1t,
                           lapack_int ldv1t, lapack_complex_double* v2t,
                           lapack_int ldv2t );

lapack_int LAPACKE_cungbr( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungbr( int matrix_order, char vect, lapack_int m,
                           lapack_int n, lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cunghr( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zunghr( int matrix_order, lapack_int n, lapack_int ilo,
                           lapack_int ihi, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cunglq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zunglq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cungql( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungql( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cungqr( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungqr( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cungrq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau );
lapack_int LAPACKE_zungrq( int matrix_order, lapack_int m, lapack_int n,
                           lapack_int k, lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau );

lapack_int LAPACKE_cungtr( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau );
lapack_int LAPACKE_zungtr( int matrix_order, char uplo, lapack_int n,
                           lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau );

lapack_int LAPACKE_cunmbr( int matrix_order, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmbr( int matrix_order, char vect, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmhr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmhr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int ilo,
                           lapack_int ihi, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmlq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmlq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmql( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmql( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmqr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmqr( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmrq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmrq( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmrz( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const lapack_complex_float* a,
                           lapack_int lda, const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmrz( int matrix_order, char side, char trans,
                           lapack_int m, lapack_int n, lapack_int k,
                           lapack_int l, const lapack_complex_double* a,
                           lapack_int lda, const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cunmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_float* a, lapack_int lda,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zunmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_double* a, lapack_int lda,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

lapack_int LAPACKE_cupgtr( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* tau,
                           lapack_complex_float* q, lapack_int ldq );
lapack_int LAPACKE_zupgtr( int matrix_order, char uplo, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* tau,
                           lapack_complex_double* q, lapack_int ldq );

lapack_int LAPACKE_cupmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_float* ap,
                           const lapack_complex_float* tau,
                           lapack_complex_float* c, lapack_int ldc );
lapack_int LAPACKE_zupmtr( int matrix_order, char side, char uplo, char trans,
                           lapack_int m, lapack_int n,
                           const lapack_complex_double* ap,
                           const lapack_complex_double* tau,
                           lapack_complex_double* c, lapack_int ldc );

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_LAPACKE_H_ */
