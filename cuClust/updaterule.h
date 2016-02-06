#ifndef _objective_h
#define _objective_h

void iterInfoCluster(thrust::device_vector<double> &d_logpci,
                     thrust::device_vector<double> &d_pci,
                     thrust::device_vector<double> &d_s_ij, 
                     thrust::device_vector<double> &d_logpc,
                     thrust::device_vector<double> &d_npc,
                     thrust::device_vector<double> &d_ps,
                     int N_c, int N, double T,
                     cublasHandle_t handle);

#endif
