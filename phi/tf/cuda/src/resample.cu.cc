#include <iostream>

#include "helpers.h"

namespace tensorflow {


// Naive CUDA kernel.
template <typename T>
__global__
void ResampleCudaKernel(
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const T* __restrict__ data,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries,
	T* __restrict__ q
) {
    q = q + (blockIdx.x * blockDim.x + threadIdx.x) * dims;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < outputSize / components; i += blockDim.x * gridDim.x){
		unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
		int n = pow2(dims);
		for (int j = 0; j < n; j++) {
			T weight = (T) 1.0;
			for (int dim = 0; dim < dims; dim++){
			    T p = ldg(points + getPointsIndex(i, dim, dims, pointsSize));
				if (checkBit(j, dim)) {
					q[dim] = floor(p) + 1;
					weight *= 1 - (q[dim] - p);
				} else {
					q[dim] = floor(p);
					weight *= 1 - (p - q[dim]);
				}
			}
			for (unsigned int component = 0; component < components; component++){
				output[i * components + component] += weight * fetchDataDevice(data, boundaries, dataBatch, q, component, dims, dimSizes, components);
			}
		}
	}
}


// https://devblogs.nvidia.com/lerp-faster-cuda/
__device__
inline float lerp (float a, float b, float x) {
	return fma(x, b, fma(-x, a, a));
}


__device__
inline float2 lerp (float2 a, float2 b, float x) {
	float2 result;
	result.x = lerp(a.x, b.x, x);
	result.y = lerp(a.y, b.y, x);
	return result;
}


__device__
inline float3 lerp (float3 a, float3 b, float x) {
	float3 result;
	result.x = lerp(a.x, b.x, x);
	result.y = lerp(a.y, b.y, x);
	result.z = lerp(a.z, b.z, x);
	return result;
}


__device__
inline float4 lerp (float4 a, float4 b, float x) {
	float4 result;
	result.x = lerp(a.x, b.x, x);
	result.y = lerp(a.y, b.y, x);
	result.z = lerp(a.z, b.z, x);
	result.w = lerp(a.w, b.w, x);
	return result;
}


// Texture memory kernel for 1D
template <typename T, typename V>
__global__
void Resample1DCudaKernel(
	const unsigned int batch,
	const unsigned int xSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		T x = ldg(points + getPointsIndex(i, 0, 1, pointsSize)); // / xSize;
		T px = floor(x);
		T fx = x - px; // fractional position

		*((V*) (output + i * components)) =  lerp(tex1DHelper<V>(dataTexture, px, boundaries, xSize), tex1DHelper<V>(dataTexture, px + 1.0, boundaries, xSize), fx);
	}
}


// Texture memory kernel for 2D
template <typename T, typename V>
__global__
void Resample2DCudaKernel (
	const unsigned int batch,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		T x = ldg(points + getPointsIndex(i, 1, 2, pointsSize));// / xSize;
		T y = ldg(points + getPointsIndex(i, 0, 2, pointsSize));// / ySize;
		T px = floor(x);
		T py = floor(y);
		T fx = x - px;	  // fractional position
		T fy = y - py;

		*((V*) (output + i * components)) = lerp(
			lerp(tex2DHelper<V>(dataTexture, px, py, boundaries, xSize, ySize),	tex2DHelper<V>(dataTexture, px + 1.0, py, boundaries, xSize, ySize), fx),
			lerp(tex2DHelper<V>(dataTexture, px, py + 1.0, boundaries, xSize, ySize), tex2DHelper<V>(dataTexture, px + 1.0, py + 1.0, boundaries, xSize, ySize), fx),
			fy
		);
	}
}


// Texture memory kernel for 3D
template <typename T, typename V>
__global__
void Resample3DCudaKernel (
	const unsigned int batch,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int zSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		T x = ldg(points + getPointsIndex(i, 2, 3, pointsSize));// / xSize;
		T y = ldg(points + getPointsIndex(i, 1, 3, pointsSize));// / ySize;
		T z = ldg(points + getPointsIndex(i, 0, 3, pointsSize));// / zSize;
		T px = floor(x);
		T py = floor(y);
		T pz = floor(z);
		T fx = x - px; // fractional position
		T fy = y - py;
		T fz = z - pz;

		*((V*) (output + i * components)) = lerp(
			lerp(
				lerp(tex3DHelper<V>(dataTexture, px, py, pz, boundaries, xSize, ySize, zSize), tex3DHelper<V>(dataTexture, px + 1.0, py, pz, boundaries, xSize, ySize, zSize), fx),
				lerp(tex3DHelper<V>(dataTexture, px, py + 1.0, pz, boundaries, xSize, ySize, zSize), tex3DHelper<V>(dataTexture, px + 1.0, py + 1.0, pz, boundaries, xSize, ySize, zSize), fx),
				fy
			),
			lerp(
				lerp(tex3DHelper<V>(dataTexture, px, py, pz + 1.0, boundaries, xSize, ySize, zSize), tex3DHelper<V>(dataTexture, px + 1.0, py, pz + 1.0, boundaries, xSize, ySize, zSize), fx),
				lerp(tex3DHelper<V>(dataTexture, px, py + 1.0, pz + 1.0, boundaries, xSize, ySize, zSize), tex3DHelper<V>(dataTexture, px + 1.0, py + 1.0, pz + 1.0, boundaries, xSize, ySize, zSize), fx),
				fy
			),
			fz
		);
	}
}


// Select kernel according to spatial rank and number of components
template<typename T>
void runResampleTextureMemoryKernel(
	const int dims,
	const unsigned int batch,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int zSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int elementsPerKernelCall,
	const unsigned int outputSize,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {
    int blockSize;
	int minGridSize;
	int gridSize;
	if(dims == 1) {
		if (components == 1) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample1DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample1DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample1DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample1DCudaKernel<float,float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample1DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		}
	} else if (dims == 2) {
		if (components == 1){
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample2DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample2DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample2DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample2DCudaKernel<float,float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample2DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		}
	} else {
		if (components == 1) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample3DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample3DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample3DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, Resample3DCudaKernel<float,float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			Resample3DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		}
	}
}


// Prepare and run texture memory kernel
template<typename T>
void ResampleTextureMemory (
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const T* __restrict__ data,
	const T* __restrict__ points,
	T* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {
	unsigned int xSize = dimSizes[dims - 1];
	unsigned int ySize = dims >= 2 ? dimSizes[dims - 2] : 0;
	unsigned int zSize = dims == 3 ? dimSizes[0] : 0;

	// Allocate cuda array
	cudaArray* cuArray = createArray(xSize, ySize, zSize, components);

	// Set 0-dimensions to 1 for size calculation
	ySize = ySize == 0 ? 1 : ySize;
	zSize = zSize == 0 ? 1 : zSize;

	// Prepare copy params for 3D
	cudaMemcpy3DParms copyParams = {0};
	if (dims == 3) {
		copyParams = createCopyParams<T>(data, cuArray, xSize, ySize, zSize, components);
	}

	// Create surface object for 3D
	cudaSurfaceObject_t surfaceObject = createSurfaceObject(cuArray);

	// Create texture object
	cudaTextureObject_t dataTexture = createTextureObject(cuArray);

	// Deal with dataBatchSize = 1 && pointsBatchSize > 1
	unsigned int elementsPerKernelCall = outputElementsPerBatch;
	if (dataBatchSize == 1 && outputSize > outputElementsPerBatch) {
		elementsPerKernelCall = outputSize;
	}

	for (unsigned int batch = 0; batch < dataBatchSize; batch++) {
		// Copy data to array
		copyDataToArray<T>(data, cuArray, surfaceObject, copyParams, dims, xSize, ySize, zSize, batch, components);

		// Run Kernel
		runResampleTextureMemoryKernel(dims, batch, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, dataTexture, points, output, boundaries);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	// Destroy texture object and free memory
	HANDLE_ERROR(cudaDestroySurfaceObject(surfaceObject));
	cudaDestroyTextureObject(dataTexture);
	HANDLE_ERROR(cudaFreeArray(cuArray));
}


// Define the GPU implementation that launches the CUDA kernel.
void LaunchResampleKernel(
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const float* __restrict__ data,
	const float* __restrict__ points,
	float* __restrict__ output,
	const Boundary* __restrict__ boundaries
) {

	// Run kernel with texture memory
	if (dims <= 3 && components <= 4){
	    if((dims == 1 && dimSizes[0] <= 8192)||
	       (dims == 2 && dimSizes[0] <= 32768 && dimSizes[1] <= 65536)||
	       (dims == 3 && dimSizes[0] <= 2048 && dimSizes[1] <= 2048 && dimSizes[2] <= 2048))
	    {
            ResampleTextureMemory<float>(
                dataBatchSize,
                dims,
                dimSizes,
                components,
                pointsSize,
                outputElementsPerBatch,
                outputSize,
                data,
                points,
                output,
                boundaries
            );
            return;
		}
	}

	// Set output field to zero
	cudaMemset(output, 0, outputSize * sizeof(float));

	int blockSize;
	int minGridSize;
	int gridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleCudaKernel<float>, 0, 0);
	gridSize = (outputSize / components + blockSize - 1) / blockSize;

	unsigned int* dimSizesDevice;
	cudaMalloc(&dimSizesDevice, dims * sizeof(unsigned int));
	cudaMemcpy(dimSizesDevice, dimSizes, dims * sizeof(unsigned int), cudaMemcpyHostToDevice);

	float* q;
	cudaMalloc(&q, gridSize * blockSize * dims * sizeof(float));

	ResampleCudaKernel<float><<<gridSize, blockSize>>>(
		dataBatchSize,
		dims,
		dimSizesDevice,
		components,
		pointsSize,
		outputElementsPerBatch,
		outputSize,
		data,
		points,
		output,
		boundaries,
		q
	);

	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(dimSizesDevice));
	HANDLE_ERROR(cudaFree(q));
}

}  // end namespace tensorflow