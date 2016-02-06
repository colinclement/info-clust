#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "arrayobject.h"

#include <cuda.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "util.h"

#define THREADS_PER 1024

__global__
void updateLogP(double *d_logpci, const double *d_logpc,
                const double *d_ps, const double *d_pci, const double *d_npc,
                const double T, const int N_c, const int N);
__global__
void subtractFromRows(double *d_logpci, const double *d_logN,
                      const int N_c, const int N);

//This function performs one iteration of the self-consistent equations
//defining a fixed point for the information-theoretic cluster probability
void iterInfoCluster(thrust::device_vector<double> &d_logpci,
                     thrust::device_vector<double> &d_pci,
                     thrust::device_vector<double> &d_s_ij, 
                     thrust::device_vector<double> &d_logpc,
                     thrust::device_vector<double> &d_npc,
                     thrust::device_vector<double> &d_ps,
                     int N_c, int N, double T,
                     cublasHandle_t handle)
{
    logsumexp(d_logpc, d_logpci, N_c, N, 1, handle);
    thrust::device_vector<double> d_ones_N(N, 1.0);
    double *d_ones_N_ptr = thrust::raw_pointer_cast(&d_ones_N[0]);
    thrust::device_vector<double> d_ones_N_c(N_c, 1.0);
    double *d_ones_N_c_ptr = thrust::raw_pointer_cast(&d_ones_N_c[0]);
    double alpha = 1.0, beta = 0.0;
    double *d_pci_ptr = thrust::raw_pointer_cast(&d_pci[0]);
    double *d_npc_ptr = thrust::raw_pointer_cast(&d_npc[0]);
    checkCudaErrors(cublasDgemv(handle, CUBLAS_OP_N, N_c, N, &alpha,
                                d_pci_ptr, N_c, d_ones_N_ptr, 1,
                                &beta, d_npc_ptr, 1));
    
    double *d_ps_ptr = thrust::raw_pointer_cast(&d_ps[0]);
    double *d_s_ij_ptr = thrust::raw_pointer_cast(&d_s_ij[0]);
    checkCudaErrors(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                N_c, N, N, &alpha, d_pci_ptr, N_c,
                                d_s_ij_ptr, N, &beta, d_ps_ptr, N_c));
    double *d_logpci_ptr = thrust::raw_pointer_cast(&d_logpci[0]);
    double *d_logpc_ptr = thrust::raw_pointer_cast(&d_logpc[0]);
    dim3 blocks((int) ceil(((float) N_c*N)/((float) THREADS_PER)), 1, 1);
    dim3 threads(THREADS_PER, 1, 1);
    updateLogP<<<blocks, threads>>>(d_logpci_ptr, d_logpc_ptr, d_ps_ptr, 
                                    d_pci_ptr, d_npc_ptr, T, N_c, N);
    //Use d_ones to normalize logpci
    logsumexp(d_ones_N, d_logpci, N_c, N, 0, handle);
    double m_alpha = -1.0;
    checkCudaErrors(cublasDger(handle, N_c, N, &m_alpha, d_ones_N_c_ptr, 1,
                               d_ones_N_ptr, 1, d_logpci_ptr, N_c));
}

//logP = logP_C[:,None]+(2*PS-P*PS/NP_C[:,None])/(T*NP_C[:,None])
__global__
void updateLogP(double *d_logpci, const double *d_logpc,
                const double *d_ps, const double *d_pci, const double *d_npc,
                const double T, const int N_c, const int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int clr = idx%N_c; //row-major ordering
    if (idx >= N_c*N)
        return;
    double ps = d_ps[idx], npc = d_npc[clr];
    d_logpci[idx] = d_logpc[clr]+(ps/T)*(2-d_pci[idx]/npc)/npc;
}






