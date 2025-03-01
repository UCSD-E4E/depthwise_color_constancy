run: build
	./build/accl.exe

build: DepthColorConsistency.cu
	nvcc -o ./build/accl .\DepthColorConsistency.cu

clean:
	del ./build/*

profile: build
	nvprof ./build/accl.exe
