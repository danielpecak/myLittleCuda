all: all1 all2

all1:
	gfortran -c vecAdd.f90 -o vecAdd.o
	nvcc  -c vecAddWrapperF.cu
	gfortran vecAddWrapperF.o vecAdd.o -o cuda_test -L/usr/local/cuda/lib64 -lcudart -lstdc++

all2:
	gfortran -c vecAdd2.f90 -o vecAdd2.o
	nvcc  -c vecAddWrapperF2.cu
	gfortran vecAddWrapperF2.o vecAdd2.o -o cuda_test2 -L/usr/local/cuda/lib64 -lcudart -lstdc++
