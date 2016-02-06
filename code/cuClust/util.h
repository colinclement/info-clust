#ifndef _cuMPF_util_h
#define _cuMPF_util_h

struct addConstant
{
    double a;
    addConstant(double c) {a = c;}

    __host__ __device__
    double operator() (double x) { return x+a;}
};

struct myExp
{
    __host__ __device__
    double operator() (double x) {return exp(x);}
};

struct myLog
{
    __host__ __device__
    double operator() (double x) {return log(x);}
};

struct myInv
{
    __host__ __device__
    double operator() (double x) {return 1./x;}
};

struct myNan_logp
{
    double scale;
    myNan_logp(double n) {scale = n;}
    
    __host__ __device__
    double operator() (double x){
        if (x == 0.0){
            return 0.0;
        } else {
            return log(scale*x);
        }
    }
};

struct myDeltaFunc
{
    __host__ __device__
    double operator() (double x){ 
        double x2 = x*x;
        return sqrt(x2*abs(1-x2)); }
};


/* Function Definitions */
void logsumexp(thrust::device_vector<double> &d_result, 
               const thrust::device_vector<double> &d_input, 
               int m, int n, int axis, cublasHandle_t handle);
void printDeviceArrayFOrder(double *d_arr, int rows, int cols);
void printDeviceArrayFOrder(int *d_arr, int rows, int cols);
int not_datatype(PyArrayObject *vec, int numpy_datatype);
void Copy_CarrayToFortranArray(double *FortranArray, double *Carray, int rows, int cols);
void Copy_CarrayToFortranArray(double *FortranArray, long int *Carray, int rows, int cols);
void Copy_FortranArrayToCarray(double *Carray, double *FortranArray, int rows, int cols);
void die(const char *message);

double infoFreeEnergy(thrust::device_vector<double> &d_pci,
                    thrust::device_vector<double> &d_s_ij,
                    double T, int N_c, int N, cublasHandle_t handle);

#endif
