/*******************************************************************************
!   Copyright(C) 1999-2013 Intel Corporation. All Rights Reserved.
!
!   The source code, information  and  material ("Material") contained herein is
!   owned  by Intel Corporation or its suppliers or licensors, and title to such
!   Material remains  with Intel Corporation  or its suppliers or licensors. The
!   Material  contains proprietary information  of  Intel or  its  suppliers and
!   licensors. The  Material is protected by worldwide copyright laws and treaty
!   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
!   modified, published, uploaded, posted, transmitted, distributed or disclosed
!   in any way  without Intel's  prior  express written  permission. No  license
!   under  any patent, copyright  or  other intellectual property rights  in the
!   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
!   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
!   intellectual  property  rights must  be express  and  approved  by  Intel in
!   writing.
!
!   *Third Party trademarks are the property of their respective owners.
!
!   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
!   this  notice or  any other notice embedded  in Materials by Intel or Intel's
!   suppliers or licensors in any way.
!
!*******************************************************************************
!  Content:
!      Intel(R) Math Kernel Library (MKL) interface for service routines
!******************************************************************************/

#ifndef _MKL_SERVICE_H_
#define _MKL_SERVICE_H_

#include <stdlib.h>
#include "mkl_types.h"

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

#if !defined(_Mkl_Api)
#define _Mkl_Api(rtype,name,arg)    extern rtype MKL_CALL_CONV name arg;
#endif

_Mkl_Api(void,MKL_Get_Version,(MKLVersion *ver)) /* Returns information about the version of the Intel MKL software */
#define mkl_get_version             MKL_Get_Version

_Mkl_Api(void,MKL_Get_Version_String,(char * buffer, int len)) /* Returns a string that contains MKL library version information */
#define mkl_get_version_string      MKL_Get_Version_String

_Mkl_Api(void,MKL_Free_Buffers,(void)) /* Frees the memory allocated by the MKL Memory Allocator */
#define mkl_free_buffers            MKL_Free_Buffers

_Mkl_Api(void,MKL_Thread_Free_Buffers,(void)) /* Frees the memory allocated by the MKL Memory Allocator in the current thread only */
#define mkl_thread_free_buffers     MKL_Thread_Free_Buffers

_Mkl_Api(MKL_INT64,MKL_Mem_Stat,(int* nbuffers)) /* MKL Memory Allocator statistical information. */
                                                 /* Returns an amount of memory, allocated by the MKL Memory Allocator */
                                                 /* in <nbuffers> buffers. */
#define mkl_mem_stat                MKL_Mem_Stat

#define  MKL_PEAK_MEM_DISABLE       0
#define  MKL_PEAK_MEM_ENABLE        1
#define  MKL_PEAK_MEM_RESET        -1
#define  MKL_PEAK_MEM               2
_Mkl_Api(MKL_INT64,MKL_Peak_Mem_Usage,(int reset))    /* Returns the peak amount of memory, allocated by the MKL Memory Allocator */
#define mkl_peak_mem_usage          MKL_Peak_Mem_Usage

_Mkl_Api(void*,MKL_malloc,(size_t size, int align)) /* Allocates the aligned buffer */
#define mkl_malloc                  MKL_malloc

_Mkl_Api(void*,MKL_calloc,(size_t num, size_t size, int align)) /* Allocates the aligned num*size - bytes memory buffer initialized by zeros */
#define mkl_calloc                  MKL_calloc

_Mkl_Api(void*,MKL_realloc,(void *ptr, size_t size)) /* Changes the size of memory buffer allocated by MKL_malloc/MKL_calloc */
#define mkl_realloc                  MKL_realloc

_Mkl_Api(void,MKL_free,(void *ptr))                 /* Frees the memory allocated by MKL_malloc() */
#define mkl_free                    MKL_free

_Mkl_Api(int,MKL_Disable_Fast_MM,(void))            /* Turns off the MKL Memory Allocator */
#define  mkl_disable_fast_mm        MKL_Disable_Fast_MM

_Mkl_Api(void,MKL_Get_Cpu_Clocks,(unsigned MKL_INT64 *)) /* Gets CPU clocks */
#define mkl_get_cpu_clocks          MKL_Get_Cpu_Clocks

_Mkl_Api(double,MKL_Get_Cpu_Frequency,(void)) /* Gets CPU frequency in GHz */
#define mkl_get_cpu_frequency       MKL_Get_Cpu_Frequency

_Mkl_Api(double,MKL_Get_Max_Cpu_Frequency,(void)) /* Gets max CPU frequency in GHz */
#define mkl_get_max_cpu_frequency   MKL_Get_Max_Cpu_Frequency

_Mkl_Api(double,MKL_Get_Clocks_Frequency,(void)) /* Gets clocks frequency in GHz */
#define mkl_get_clocks_frequency    MKL_Get_Clocks_Frequency

_Mkl_Api(int,MKL_Set_Num_Threads_Local,(int nth))
#define mkl_set_num_threads_local   MKL_Set_Num_Threads_Local
_Mkl_Api(void,MKL_Set_Num_Threads,(int nth))
#define mkl_set_num_threads         MKL_Set_Num_Threads
_Mkl_Api(int,MKL_Get_Max_Threads,(void))
#define mkl_get_max_threads         MKL_Get_Max_Threads
_Mkl_Api(int,MKL_Domain_Set_Num_Threads,(int nth, int MKL_DOMAIN))
#define mkl_domain_set_num_threads  MKL_Domain_Set_Num_Threads
_Mkl_Api(int,MKL_Domain_Get_Max_Threads,(int MKL_DOMAIN))
#define mkl_domain_get_max_threads  MKL_Domain_Get_Max_Threads
_Mkl_Api(void,MKL_Set_Dynamic,(int bool_MKL_DYNAMIC))
#define mkl_set_dynamic             MKL_Set_Dynamic
_Mkl_Api(int,MKL_Get_Dynamic,(void))
#define mkl_get_dynamic             MKL_Get_Dynamic

/* MKL Progress routine */
#ifndef _MKL_PROGRESS_H_
#define _MKL_PROGRESS_H_
_Mkl_Api(int,MKL_PROGRESS, ( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,MKL_PROGRESS_,( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,mkl_progress, ( int* thread, int* step, char* stage, int lstage ))
_Mkl_Api(int,mkl_progress_,( int* thread, int* step, char* stage, int lstage ))
#endif /* _MKL_PROGRESS_H_ */

_Mkl_Api(int,MKL_Enable_Instructions,(int))
#define  mkl_enable_instructions    MKL_Enable_Instructions
#define  MKL_AVX_ENABLE             1
#define  MKL_SINGLE_PATH_ENABLE     0x0600

/* Single Dynamic library interface */
#define MKL_INTERFACE_LP64          0
#define MKL_INTERFACE_ILP64         1
_Mkl_Api(int,MKL_Set_Interface_Layer,(int code))
#define mkl_set_interface_layer     MKL_Set_Interface_Layer

/* Single Dynamic library threading */
#define MKL_THREADING_INTEL         0
#define MKL_THREADING_SEQUENTIAL    1
#define MKL_THREADING_PGI           2
_Mkl_Api(int,MKL_Set_Threading_Layer,(int code))
#define mkl_set_threading_layer     MKL_Set_Threading_Layer

typedef void (* XerblaEntry) (char * Name, int * Num, int Len);
_Mkl_Api(XerblaEntry,mkl_set_xerbla,(XerblaEntry xerbla))

typedef int (* ProgressEntry) (int* thread, int* step, char* stage, int stage_len);
_Mkl_Api(ProgressEntry,mkl_set_progress,(ProgressEntry progress))

/* MIC service routines */
_Mkl_Api(int,MKL_MIC_Enable,(void))
#define mkl_mic_enable              MKL_MIC_Enable
_Mkl_Api(int,MKL_MIC_Disable,(void))
#define mkl_mic_disable             MKL_MIC_Disable

_Mkl_Api(int,MKL_MIC_Get_Device_Count,(void))
#define mkl_mic_get_device_count MKL_MIC_Get_Device_Count

typedef enum MKL_MIC_TARGET_TYPE {
    MKL_TARGET_NONE = 0, /* Undefine target */
    MKL_TARGET_HOST = 1, /* Host used as target */
    MKL_TARGET_MIC  = 2  /* MIC target */
} MKL_MIC_TARGET_TYPE;

#define MKL_MIC_DEFAULT_TARGET_TYPE MKL_TARGET_MIC
#define MKL_MIC_DEFAULT_TARGET_NUMBER 0
#define MKL_MIC_AUTO_WORKDIVISION   -1.0

_Mkl_Api(int,MKL_MIC_Set_Workdivision,(MKL_MIC_TARGET_TYPE target_type,
                                       int target_number, double wd))
#define mkl_mic_set_workdivision    MKL_MIC_Set_Workdivision

_Mkl_Api(int,MKL_MIC_Get_Workdivision,(MKL_MIC_TARGET_TYPE target_type,
                                       int target_number, double *wd))
#define mkl_mic_get_workdivision    MKL_MIC_Get_Workdivision

_Mkl_Api(int,MKL_MIC_Set_Max_Memory,(MKL_MIC_TARGET_TYPE target_type,
                                     int target_number, size_t card_mem_mbytes))
#define mkl_mic_set_max_memory      MKL_MIC_Set_Max_Memory

_Mkl_Api(int,MKL_MIC_Free_Memory,(MKL_MIC_TARGET_TYPE target_type,
                                  int target_number))
#define mkl_mic_free_memory         MKL_MIC_Free_Memory

_Mkl_Api(int,MKL_MIC_Set_Offload_Report,(int enabled))
#define mkl_mic_set_offload_report  MKL_MIC_Set_Offload_Report

_Mkl_Api(int,MKL_MIC_Set_Device_Num_Threads,(MKL_MIC_TARGET_TYPE target_type,
                                             int target_number, int num_threads))
#define mkl_mic_set_device_num_threads MKL_MIC_Set_Device_Num_Threads


/* MKL CBWR */
_Mkl_Api(int,MKL_CBWR_Get,(int))
#define mkl_cbwr_get                MKL_CBWR_Get
_Mkl_Api(int,MKL_CBWR_Set,(int))
#define mkl_cbwr_set                MKL_CBWR_Set
_Mkl_Api(int,MKL_CBWR_Get_Auto_Branch,(void))
#define mkl_cbwr_get_auto_branch    MKL_CBWR_Get_Auto_Branch

/* Obsolete */
_Mkl_Api(void,MKL_Set_Cpu_Frequency,(double*))
#define mkl_set_cpu_frequency       MKL_Set_Cpu_Frequency

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_SERVICE_H_ */
