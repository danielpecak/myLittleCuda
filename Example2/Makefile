all: all1 

all1:
	gfortran -c matMul.f90 -o matMul.o
	nvcc  -c matMulWrapperF.cu
	gfortran matMulWrapperF.o matMul.o -o cuda_test -L/usr/local/cuda/lib64 -lcudart -lstdc++

all2:
	gfortran -c matMul2.f90 -o matMul2.o
	nvcc  -c matMulWrapperF2.cu
	gfortran matMulWrapperF2.o matMul2.o -o cuda_test2 -L/usr/local/cuda/lib64 -lcudart -lstdc++
