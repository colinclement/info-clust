PYTHON_DIR = $(HOME)/anaconda
CUDA_INC = -I$(CUDA_HOME)/inc
CUDA_LIB = -L$(CUDA_HOME)/lib64 -lcudart -lcublas
OPENCV_INC = -I/usr/local/include
OPENCV_LIB = -L/usr/local/lib -lopencv_core -lopencv_cudaarithm

#Numpy headers
INC = -I$(PYTHON_DIR)/include/python2.7/
INC += -I$(PYTHON_DIR)/lib/python2.7/site-packages/numpy/core/include/numpy/

#CUDA helper function for debugging
INC += -I$(CUDA_HOME)/samples/common/inc
INC += -I$(HOME)/include

CFLAGS = -O3 #-g 
CUDA_FLAGS = #--use_fast_math

all: _cuClust.so

_cuClust.so: cuClust.o util.o updaterule.o
	g++ $(CFLAGS) -fPIC -shared -o _cuClust.so cuClust.o updaterule.o util.o $(CUDA_LIB) $(OPENCV_LIB)

cuClust.o: cuClust.cu
	nvcc $(CFLAGS) $(CUDA_FLAGS) -c cuClust.cu -Xcompiler -fPIC $(INC) $(CUDA_INC)

updaterule.o: updaterule.cu 
	nvcc $(CFLAGS) $(CUDA_FLAGS) -c updaterule.cu -Xcompiler -fPIC $(INC) $(CUDA_INC) $(LIB)
	
util.o: util.cu
	nvcc $(CFLAGS) $(CUDA_FLAGS) -c util.cu -Xcompiler -fPIC $(INC) $(CUDA_INC) $(OPENCV_INC)

clean:
	rm -f *.o *.so
