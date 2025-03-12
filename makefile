run: build
	./build/accl

build: DepthColorConsistency.cu
	nvcc -g -lcublas -o ./build/accl DepthColorConsistency.cu

clean:
	del ./build/*

profile: build
	nvprof ./build/accl.exe 
