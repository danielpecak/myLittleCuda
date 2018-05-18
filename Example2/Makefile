all: matmul_naive

matmul_naive: matMul_naive.o matMulWrapperF.o
	gfortran $^ -o $@ -L/usr/local/cuda/lib64 -lcudart -lstdc++

matMul_naive.o: matMul_naive.f90
	gfortran -c $< -o $@

matMulWrapperF.o: matMulWrapperF.cu
	nvcc  -c $<


all2:
	gfortran -c matMul2.f90 -o matMul2.o
	nvcc  -c matMulWrapperF2.cu
	gfortran matMulWrapperF2.o matMul2.o -o cuda_test2 -L/usr/local/cuda/lib64 -lcudart -lstdc++

clean:
	rm -f *.o *.mod cuda_test cuda_test2 matmul_naive
