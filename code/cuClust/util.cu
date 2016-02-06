#include "Python.h"
#include <stdlib.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "arrayobject.h"
#include <errno.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include "util.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"


void logsumexp(thrust::device_vector<double> &d_result, 
               const thrust::device_vector<double> &d_input, 
               int m, int n, int axis, cublasHandle_t handle)
{

    using namespace thrust::placeholders;
    cublasOperation_t op = CUBLAS_OP_T;
    int ones_size = m;

    if (axis){ // Sum over columns
        op = CUBLAS_OP_N;
        ones_size = n;
    } 

    thrust::device_vector<double> d_expin = d_input;
    const double *d_input_ptr = thrust::raw_pointer_cast(&d_input[0]);
    double *d_expin_ptr = thrust::raw_pointer_cast(&d_expin[0]);
    double *d_result_ptr = thrust::raw_pointer_cast(&d_result[0]);
    thrust::device_vector<double> d_ones(ones_size, 1.0);
    double *d_ones_ptr = thrust::raw_pointer_cast(&d_ones[0]);
    
    double alpha = 1.0, beta = 0.0;
    
    cv::cuda::GpuMat CVmat_input(n, m, CV_64FC1, (void *) d_input_ptr);
    cv::cuda::GpuMat d_maxima;

    double *d_maxima_ptr = NULL;
    double m_alpha = -1.0;
    if (axis){ //subtract max from rows
        cv::cuda::reduce(CVmat_input, d_maxima, 0, CV_REDUCE_MAX, -1);
        d_maxima_ptr = (double *)d_maxima.ptr();
        checkCudaErrors(cublasDger(handle, m, n, &m_alpha, d_maxima_ptr,
                                   1, d_ones_ptr, 1, d_expin_ptr, m));
    } else { // subtract max from columns
        cv::cuda::reduce(CVmat_input, d_maxima, 1, CV_REDUCE_MAX, -1);
        d_maxima_ptr = (double *)d_maxima.ptr();
        checkCudaErrors(cublasDger(handle, m, n, &m_alpha, d_ones_ptr,
                                   1, d_maxima_ptr, 1, d_expin_ptr, m));
    }
    thrust::device_ptr<double> d_maxima_thrust = thrust::device_pointer_cast(d_maxima_ptr);
    //thrust::host_vector<double> h_maxima(d_maxima_thrust, d_maxima_thrust+sums_size);
    //printf("Maximum values:\n");
    //for (int i=0; i<h_maxima.size(); i++){
    //    printf("%f ", h_maxima[i]);
    //}
    //printf("\n")

    myExp expy;
    myLog logy;
    thrust::transform(d_expin.begin(), d_expin.end(), d_expin.begin(), expy);
    
    checkCudaErrors(cublasDgemv(handle, op, m, n, &alpha, 
                                d_expin_ptr, m, d_ones_ptr, 1,
                                &beta, d_result_ptr, 1));
    
    thrust::transform(d_result.begin(), d_result.end(), d_result.begin(), logy);
    thrust::transform(d_result.begin(), d_result.end(), d_maxima_thrust,
                      d_result.begin(), thrust::plus<double>());
}

double infoFreeEnergy(thrust::device_vector<double> &d_pci,
                    thrust::device_vector<double> &d_s_ij,
                    double T, int N_c, int N, cublasHandle_t handle)
{
    double *d_pci_ptr = thrust::raw_pointer_cast(&d_pci[0]);
    double *d_s_ij_ptr = thrust::raw_pointer_cast(&d_s_ij[0]);
    thrust::device_vector<double> d_npc(N_c);
    double *d_npc_ptr = thrust::raw_pointer_cast(&d_npc[0]);
    thrust::device_vector<double> d_ps(N_c*N);
    double *d_ps_ptr = thrust::raw_pointer_cast(&d_ps[0]);
    thrust::device_vector<double> d_ones_N(N, 1.0);
    double *d_ones_N_ptr = thrust::raw_pointer_cast(&d_ones_N[0]);

    double alpha = 1.0, beta = 0.0;
    checkCudaErrors(cublasDgemv(handle, CUBLAS_OP_N, N_c, N, &alpha,
                                d_pci_ptr, N_c, d_ones_N_ptr, 1,
                                &beta, d_npc_ptr, 1));

    checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                N_c, N, N, &alpha, d_pci_ptr, N_c,
                                d_s_ij_ptr, N, &beta, d_ps_ptr, N_c));

    thrust::device_vector<double> d_pci_by_npc(N_c*N);
    double *d_pci_by_npc_ptr = thrust::raw_pointer_cast(&d_pci_by_npc[0]);

    //rescale rows
    myInv invy; 
    thrust::transform(d_npc.begin(), d_npc.end(), d_npc.begin(), invy);
    checkCudaErrors(cublasDdgmm(handle, CUBLAS_SIDE_LEFT, N_c, N, 
                                d_pci_ptr, N_c, d_npc_ptr, 1, 
                                d_pci_by_npc_ptr, N_c));
    
    double S=0;
    checkCudaErrors(cublasDdot(handle, N_c*N, d_pci_by_npc_ptr,
                               1, d_ps_ptr, 1, &S));
    
    myNan_logp nanLogp((double)N);
    thrust::transform(d_pci_by_npc.begin(), d_pci_by_npc.end(), 
                      d_pci_by_npc.begin(), nanLogp);

    double I=0;
    checkCudaErrors(cublasDdot(handle, N_c*N, d_pci_ptr,
                               1, d_pci_by_npc_ptr, 1, &I));

    return (S-T*I)/N;
}




void printDeviceArrayFOrder(double *d_arr, int rows, int cols)
{
    int BYTES = sizeof(double)*rows*cols;
    double *h_arr = (double *)malloc(BYTES);
    checkCudaErrors(cudaMemcpy(h_arr, d_arr, BYTES, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i< rows; i++){
        for (int j=0; j<cols; j++){
            printf("%f ", h_arr[j*rows+i]);
        }
        printf("\n");
    }
    free(h_arr);
}

void printDeviceArrayFOrder(int *d_arr, int rows, int cols)
{
    int BYTES = sizeof(int)*rows*cols;
    int *h_arr = (int *)malloc(BYTES);
    checkCudaErrors(cudaMemcpy(h_arr, d_arr, BYTES, cudaMemcpyDeviceToHost));
    
    for (int i = 0; i< rows; i++){
        for (int j=0; j<cols; j++){
            printf("%i ", h_arr[j*rows+i]);
        }
        printf("\n");
    }
    free(h_arr);
}

/* Check that PyArrayObject is a double (Float) type and a vector */ 
int not_datatype(PyArrayObject *vec, int numpy_dtype)  
{
   if (PyArray_DESCR(vec)->type_num != numpy_dtype)  {
      PyErr_SetString(PyExc_ValueError,
                      "Array is incorrect datatype");
      return 1;  
   }
   return 0;
}

void die(const char *message)
{
    if (errno){
        perror(message);
    } else {
        printf("ERROR: %s\n", message);
    }
}

void Copy_CarrayToFortranArray(double * FortranArray, double *Carray, 
                               int row, int col)
{
    for (int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            FortranArray[j*row+i] = Carray[i*col+j];
        }
    }
}

//Overload for ints too, casts int to double!!
void Copy_CarrayToFortranArray(double * FortranArray, long int *Carray, 
                               int row, int col)
{
    for (int i = 0; i<row; i++){
        for (int j = 0; j<col; j++){
            FortranArray[j*row+i] = (double)Carray[i*col+j];
        }
    }
}

void Copy_FortranArrayToCarray(double *Carray, double *FortranArray, 
                               int row, int col)
{
    for (int i = 0; i<row; i++){
        for (int j= 0; j<col; j++){
            Carray[i*col+j] = FortranArray[j*row+i];
        }
    }
}


