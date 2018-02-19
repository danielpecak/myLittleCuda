all: 
	gfortran -c vecAdd.f90 -o vecAdd.o
	nvcc  -c vecAddWrapperF.cu
	gfortran vecAddWrapperF.o vecAdd.o -o cuda_test -L/usr/local/cuda/lib64 -lcudart -lstdc++


