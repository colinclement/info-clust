#include "Python.h"
#include <stdlib.h>
#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION
#include "arrayobject.h"
#include <math.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <helper_cuda.h> //In samples/common/inc
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include "util.h"
#include "updaterule.h"

#define CUDA_DEVICE 0

// Python interface
static PyObject *optimizeP(PyObject *self, PyObject *args);
extern "C" void init_cuClust();

/* Python Module Methods */
static PyMethodDef _MethodDefcuClust[] = {
    {"optimizeP", optimizeP, METH_VARARGS},
    {NULL, NULL, 0}
};

/* Initialize C Extension */
void init_cuClust()  
{
    (void) Py_InitModule("_cuClust", _MethodDefcuClust);
    import_array();
}

static PyObject *optimizeP(PyObject *self, PyObject *args)
{
    
    PyArrayObject *s_ij, *pci;
    double T, eps;
    int maxiter, iprint = 0;
    /* Parse parameters and verify input */
    if (!PyArg_ParseTuple(args, "O!O!ddii", &PyArray_Type, &s_ij, 
                          &PyArray_Type, &pci,
                          &T, &eps, &maxiter, &iprint)) {
        die("Failed to parse python argument\n");
        return NULL;
    }

    if (NULL == s_ij || not_datatype(s_ij, NPY_FLOAT64) ){
        die("s_ij is not of datatype float\n");
        return NULL;
    }
    if (NULL == pci || not_datatype(pci, NPY_FLOAT64) ){
        die("P_C_i is not of datatype float\n");
        return NULL;
    }
    if ( PyArray_DIM(s_ij, 0) != PyArray_DIM(s_ij, 1)){
        die("s_ij is not square\n");
        return NULL;
    }
    if (PyArray_DIM(pci, 1) != PyArray_DIM(s_ij, 0)){
        die("P_C_i and s_ij are imcompatible shapes\n");
        return NULL;
    }

    int N_c = PyArray_DIM(pci, 0);
    int N = PyArray_DIM(s_ij, 0);

    // Set up CUDA
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(!deviceCount){
        fprintf(stderr, "Error: No CUDA supporting devices.\n");
        exit(EXIT_FAILURE);
    }
    cudaSetDevice(CUDA_DEVICE);
    
    cublasHandle_t handle;
    checkCudaErrors(cublasCreate(&handle));

    //Copy data to device
    double *pci_ptr = (double *)PyArray_DATA(pci);
    thrust::host_vector<double> h_pci_f(N_c*N);
    double *pci_f_ptr = thrust::raw_pointer_cast(&h_pci_f[0]);
    Copy_CarrayToFortranArray(pci_f_ptr, pci_ptr, N_c, N);
    thrust::device_vector<double> d_pci = h_pci_f;
    
    double *s_ij_ptr = (double *)PyArray_DATA(s_ij);
    thrust::device_vector<double> d_s_ij(s_ij_ptr, s_ij_ptr+N*N);

    using namespace thrust::placeholders;

    //Create working memory 
    thrust::device_vector<double> d_logpci(N_c*N);
    myLog logy;
    myExp expy;
    thrust::transform(d_pci.begin(), d_pci.end(), d_logpci.begin(), logy);
    thrust::device_vector<double> d_logpc(N_c, 1.), d_npc(N_c, 1.), d_ps(N_c*N, 1.);
    thrust::device_vector<double> d_deltaP(N_c*N, 1.);
    thrust::device_vector<double> d_oldpci(N_c*N);

    int finished = 0, iter = 0;

    if (iprint){
        double F = infoFreeEnergy(d_pci, d_s_ij, T, N_c, N, handle);
        printf("Initial F = %f\n", F);
    }

    while (!finished){
        iter += 1;
        thrust::copy(d_pci.begin(), d_pci.end(), d_oldpci.begin());

        iterInfoCluster(d_logpci, d_pci, d_s_ij, d_logpc, d_npc, d_ps,
                        N_c, N, T, handle);

        thrust::transform(d_logpci.begin(), d_logpci.end(), d_pci.begin(), expy);
    
        //Test for convergence
        thrust::transform(d_pci.begin(), d_pci.end(), d_oldpci.begin(),
                          d_deltaP.begin(), thrust::minus<double>());
        myDeltaFunc dfunc;
        double deltaP = thrust::transform_reduce(d_deltaP.begin(), d_deltaP.end(),
                                                 dfunc, 0.0, thrust::plus<double>())/(N*N_c);

        if (iprint>1){
            printf("\t%i: deltaP = %E\n", iter, deltaP);
        }

        if (deltaP < eps)
            finished = 1;
        if (iter >= maxiter){
            finished = 1;
            printf("%i (Maximum) iterations reached\n", maxiter);
        }
    }
    
    if (iprint){
        printf("Finished in %i iterations\n", iter);
    }
    
    if (iprint){
        double F = infoFreeEnergy(d_pci, d_s_ij, T, N_c, N, handle);
        printf("Final F = %f\n", F);
    }

    thrust::transform(d_logpci.begin(), d_logpci.end(), d_pci.begin(), expy);

    //Copy back to P passed in
    h_pci_f = d_pci;
    Copy_FortranArrayToCarray(pci_ptr, pci_f_ptr, N_c, N);

    //PyArrayObject *logpc = (PyArrayObject *)PyArray_FromDims(1, &N_c, NPY_DOUBLE);
    //PyArrayObject *npc = (PyArrayObject *)PyArray_FromDims(1, &N_c, NPY_DOUBLE);
    //int Pshape[2] = {N_c, N};
    //PyArrayObject *ps = (PyArrayObject *)PyArray_FromDims(2, Pshape, NPY_DOUBLE);
    //PyArrayObject *logpci = (PyArrayObject *)PyArray_FromDims(2, Pshape, NPY_DOUBLE);
    //PyArrayObject *oldpci = (PyArrayObject *)PyArray_FromDims(2, Pshape, NPY_DOUBLE);
    //double *ps_f = (double *)malloc(N_c*N*sizeof(double));
    //double *logpci_f = (double *)malloc(N_c*N*sizeof(double));
    //double *oldpci_f = (double *)malloc(N_c*N*sizeof(double));
    //
    //double *d_logpc_ptr = thrust::raw_pointer_cast(&d_logpc[0]);
    //double *d_npc_ptr = thrust::raw_pointer_cast(&d_npc[0]);
    //double *d_ps_ptr = thrust::raw_pointer_cast(&d_ps[0]);
    //double *d_logpci_ptr = thrust::raw_pointer_cast(&d_logpci[0]);
    //double *d_oldpci_ptr = thrust::raw_pointer_cast(&d_oldpci[0]);
    //double *h_logpc_ptr = (double *)PyArray_DATA(logpc);
    //checkCudaErrors(cudaMemcpy(h_logpc_ptr, d_logpc_ptr,
    //                           sizeof(double)*N_c, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy((double *)PyArray_DATA(npc), d_npc_ptr,
    //                           sizeof(double)*N_c, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(ps_f, d_ps_ptr,
    //                           sizeof(double)*N_c*N, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(logpci_f, d_logpci_ptr,
    //                           sizeof(double)*N_c*N, cudaMemcpyDeviceToHost));
    //checkCudaErrors(cudaMemcpy(oldpci_f, d_oldpci_ptr,
    //                           sizeof(double)*N_c*N, cudaMemcpyDeviceToHost));
    //Copy_FortranArrayToCarray((double *)PyArray_DATA(ps), ps_f, N_c, N); 
    //Copy_FortranArrayToCarray((double *)PyArray_DATA(logpci), logpci_f, N_c, N); 
    //Copy_FortranArrayToCarray((double *)PyArray_DATA(oldpci), oldpci_f, N_c, N); 
    //free(ps_f);
    //free(logpci_f);

    //Grab final errors
    checkCudaErrors(cublasDestroy(handle));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    //Build return value
    //PyObject * tupleresult = PyTuple_New(3);
    //PyTuple_SetItem(tupleresult, 1, PyArray_Return(logpci));
    //PyTuple_SetItem(tupleresult, 2, PyArray_Return(oldpci));
    //PyTuple_SetItem(tupleresult, 0, Py_BuildValue("i", iter));
    //return tupleresult;
    return Py_BuildValue("i", iter);
}








