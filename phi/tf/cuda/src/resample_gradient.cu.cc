#include <iostream>
#include <math.h>// nextafter()
#include "helpers.h"


namespace tensorflow {




// Naive CUDA kernel
template <typename T>
__global__
void ResampleGradientCudaKernel(
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	const T* __restrict__ data,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries,
	T* __restrict__ q,
	T* __restrict__ weights
) {
    unsigned int absThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    q = q + absThreadIdx * dims;
    weights = weights + absThreadIdx * (dims + 1);
	for (unsigned int i = absThreadIdx; i < outputSize / components; i += blockDim.x * gridDim.x){
		unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
		int n = pow2(dims);
		for (int j = 0; j < n; j++) {
			for (int dim = 0; dim <= dims; dim++){
				weights[dim] = 1;
			}
			for (int dim = 0; dim < dims; dim++){
			    T p = ldg(points + getPointsIndex(i, dim, dims, pointsSize));
				T weight = 1;
				if (checkBit(j, dim)) {
					q[dim] = floor(p) + 1;
					weight = 1 - (q[dim] - p);
				} else {
					q[dim] = floor(p);
					weight = 1 - (p - q[dim]);
				}
				for (int otherDim = 0; otherDim <= dims; otherDim++){
					if(otherDim != dim) {
						weights[otherDim] *= weight;
					}
				}
			}
			for (unsigned int component = 0; component < components; component++) {
				T dataValue = 0;
				T outputGradientValue = ldg(outputGradient + ((i * components + component) % outputGradientSize));
				if (applyBoundaries(boundaries, q, dims, dimSizes)) {
					// compute dataGradient
					unsigned int dataIndex = getDataIndex(dataBatch, q, component, dims, dimSizes, components);
					atomicAdd(dataGradient + dataIndex, weights[dims] * outputGradientValue);
					dataValue = ldg(data + dataIndex);
				}

				// compute pointsGradient
				for (int dim = 0; dim < dims; dim++) {
					int factor = checkBit(j, dim) ? 1 : -1;
					atomicAdd(pointsGradient + getPointsIndex(i, dim, dims, pointsSize), factor * weights[dim] * dataValue * outputGradientValue);
				}
			}
		}
	}
}


// Texture memory kernel for 1D
template <typename T, typename V>
__global__
void ResampleGradient1DCudaKernel(
	const unsigned int batch,
	const unsigned int dataBatchSize,
	const unsigned int xSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
		T x = ldg(points + getPointsIndex(i, 0, 1, pointsSize));
		T pa = floor(x);
		T pb = pa + 1;

		T fx = x - pa; // fractional position

		V a;
		bool aExists = applyBoundaries(boundaries, &pa, 1, &xSize);
		if (aExists) {
			a = tex1DHelper<V>(dataTexture, pa + 0.5);
		} else {
			memset(&a, 0, sizeof(V));
		}

		V b;
		bool bExists = applyBoundaries(boundaries, &pb, 1, &xSize);
		if (bExists) {
			b = tex1DHelper<V>(dataTexture, pb + 0.5);
		} else {
			memset(&b, 0, sizeof(V));
		}

		for (int component = 0; component < components; component++){
			T outputGrad = ldg(outputGradient + ((i * components + component) % outputGradientSize));
			// compute dataGradient
			if(aExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, &pa, component, 1, &xSize, components), (1 - fx) * outputGrad);
			if(bExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, &pb, component, 1, &xSize, components), fx * outputGrad);

			// compute PointsGradient
			pointsGradient[getPointsIndex(i, 0, 1, pointsSize)] += (*(((float*) &b) + component) - *(((float*) &a) + component)) * outputGrad;
		}
	}
}


// Texture memory kernel for 2D
template <typename T, typename V>
__global__
void ResampleGradient2DCudaKernel (
	const unsigned int batch,
	const unsigned int dataBatchSize,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
		unsigned int dimSizes[2] = {ySize, xSize};

		T y = ldg(points + getPointsIndex(i, 0, 2, pointsSize));
		T x = ldg(points + getPointsIndex(i, 1, 2, pointsSize));

		T pa[2] = {(T) floor(y), (T) floor(x)};
		T pb[2] = {pa[0], pa[1] + 1};
		T pc[2] = {pa[0] + 1, pa[1]};
		T pd[2] = {pc[0], pb[1]};

		T fy = y - pa[0]; // fractional position
		T fx = x - pa[1];

		// a
		V a;
		bool aExists = applyBoundaries(boundaries, pa, 2, dimSizes);
		if (aExists) {
			a = tex2DHelper<V>(dataTexture, pa[1] + 0.5, pa[0] + 0.5);
		} else {
			memset(&a, 0, sizeof(V));
		}
		// b
		V b;
		bool bExists = applyBoundaries(boundaries, pb, 2, dimSizes);
		if (bExists) {
			b = tex2DHelper<V>(dataTexture, pb[1] + 0.5, pb[0] + 0.5);
		} else {
			memset(&b, 0, sizeof(V));
		}
		// c
		V c;
		bool cExists = applyBoundaries(boundaries, pc, 2, dimSizes);
		if (cExists) {
			c = tex2DHelper<V>(dataTexture, pc[1] + 0.5, pc[0] + 0.5);
		} else {
			memset(&c, 0, sizeof(V));
		}
		// d
		V d;
		bool dExists = applyBoundaries(boundaries, pd, 2, dimSizes);
		if (dExists) {
			d = tex2DHelper<V>(dataTexture, pd[1] + 0.5, pd[0] + 0.5);
		} else {
			memset(&a, 0, sizeof(V));
		}

		for (int component = 0; component < components; component++){
			T outputGrad = ldg(outputGradient + ((i * components + component) % outputGradientSize));
			// compute dataGradient
			if (aExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pa, component, 2, dimSizes, components), (1 - fx) * (1 - fy) * outputGrad);
			if (bExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pb, component, 2, dimSizes, components), fx * (1 - fy) * outputGrad);
			if (cExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pc, component, 2, dimSizes, components), (1 - fx) * fy * outputGrad);
			if (dExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pd, component, 2, dimSizes, components), fx * fy * outputGrad);

			// compute pointsGradient
			T aComp = *(((T*) &a) + component);
			T bComp = *(((T*) &b) + component);
			T cComp = *(((T*) &c) + component);
			T dComp = *(((T*) &d) + component);
			pointsGradient[getPointsIndex(i, 0, 2, pointsSize)] += (fx * dComp + (1 - fx) * cComp - fx * bComp + (fx - 1) * aComp) * outputGrad; // y
			pointsGradient[getPointsIndex(i, 1, 2, pointsSize)] += (fy * dComp - fy * cComp + (1 - fy) * bComp + (fy - 1) * aComp) * outputGrad; // x
		}
	}
}


// Texture memory kernel for 3D
template <typename T, typename V>
__global__
void ResampleGradient3DCudaKernel (
	const unsigned int batch,
	const unsigned int dataBatchSize,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int zSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
) {
	for (unsigned int i = batch * outputElementsPerBatch / components + blockIdx.x * blockDim.x + threadIdx.x; i < (batch * outputElementsPerBatch + outputElementsPerBatch) / components; i += blockDim.x * gridDim.x){
		unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
		unsigned int dimSizes[3] = {zSize, ySize, xSize};

		T z = ldg(points + getPointsIndex(i, 0, 3, pointsSize));
		T y = ldg(points + getPointsIndex(i, 1, 3, pointsSize));
		T x = ldg(points + getPointsIndex(i, 2, 3, pointsSize));

		T pa[3] = {(T) floor(z), (T) floor(y), (T) floor(x)};
		T pb[3] = {pa[0], pa[1], pa[2] + 1};
		T pc[3] = {pa[0], pa[1] + 1, pa[2]};
		T pd[3] = {pa[0], pc[1], pb[2]};
		T pe[3] = {pa[0] + 1, pa[1], pa[2]};
		T pf[3] = {pe[0], pa[1], pb[2]};
		T pg[3] = {pe[0], pc[1], pa[2]};
		T ph[3] = {pe[0], pc[1], pb[2]};

		T fz = z - pa[0]; // fractional position
		T fy = y - pa[1];
		T fx = x - pa[2];

		// a
		V a;
		bool aExists = applyBoundaries(boundaries, pa, 3, dimSizes);
		if (aExists) {
			a = tex3DHelper<V>(dataTexture, pa[2], pa[1], pa[0]);
		} else {
			memset(&a, 0, sizeof(V));
		}
		// b
		V b;
		bool bExists = applyBoundaries(boundaries, pb, 3, dimSizes);
		if (bExists) {
			b = tex3DHelper<V>(dataTexture, pb[2], pb[1], pb[0]);
		} else {
			memset(&b, 0, sizeof(V));
		}
		// c
		V c;
		bool cExists = applyBoundaries(boundaries, pc, 3, dimSizes);
		if (cExists) {
			c = tex3DHelper<V>(dataTexture, pc[2], pc[1], pc[0]);
		} else {
			memset(&c, 0, sizeof(V));
		}
		// d
		V d;
		bool dExists = applyBoundaries(boundaries, pd, 3, dimSizes);
		if (dExists) {
			d = tex3DHelper<V>(dataTexture, pd[2], pd[1], pd[0]);
		} else {
			memset(&d, 0, sizeof(V));
		}
		// e
		V e;
		bool eExists = applyBoundaries(boundaries, pe, 3, dimSizes);
		if (eExists) {
			e = tex3DHelper<V>(dataTexture, pe[2], pe[1], pe[0]);
		} else {
			memset(&e, 0, sizeof(V));
		}
		// f
		V f;
		bool fExists = applyBoundaries(boundaries, pf, 3, dimSizes);
		if (fExists) {
			f = tex3DHelper<V>(dataTexture, pf[2], pf[1], pf[0]);
		} else {
			memset(&f, 0, sizeof(V));
		}
		// g
		V g;
		bool gExists = applyBoundaries(boundaries, pg, 3, dimSizes);
		if (gExists) {
			g = tex3DHelper<V>(dataTexture, pg[2], pg[1], pg[0]);
		} else {
			memset(&g, 0, sizeof(V));
		}
		// h
		V h;
		bool hExists = applyBoundaries(boundaries, ph, 3, dimSizes);
		if (hExists) {
			h = tex3DHelper<V>(dataTexture, ph[2], ph[1], ph[0]);
		} else {
			memset(&h, 0, sizeof(V));
		}

		for (int component = 0; component < components; component++){
			T outputGrad = ldg(outputGradient + ((i * components + component) % outputGradientSize));
			// compute dataGradient
			if (aExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pa, component, 3, dimSizes, components), (1 - fx) * (1 - fy) * (1 - fz) * outputGrad);
			if (bExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pb, component, 3, dimSizes, components), fx * (1 - fy) * (1 - fz) * outputGrad);
			if (cExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pc, component, 3, dimSizes, components), (1 - fx) * fy * (1 - fz) * outputGrad);
			if (dExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pd, component, 3, dimSizes, components), fx * fy * (1 - fz) * outputGrad);
			if (eExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pe, component, 3, dimSizes, components), (1 - fx) * (1 - fy) * fz * outputGrad);
			if (fExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pf, component, 3, dimSizes, components), fx * (1 - fy) * fz * outputGrad);
			if (gExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, pg, component, 3, dimSizes, components), (1 - fx) * fy * fz * outputGrad);
			if (hExists)
				atomicAdd(dataGradient + getDataIndex(dataBatch, ph, component, 3, dimSizes, components), fx * fy * fz * outputGrad);

			// compute pointsGradient
			T aComp = *(((T*) &a) + component);
			T bComp = *(((T*) &b) + component);
			T cComp = *(((T*) &c) + component);
			T dComp = *(((T*) &d) + component);
			T eComp = *(((T*) &e) + component);
			T fComp = *(((T*) &f) + component);
			T gComp = *(((T*) &g) + component);
			T hComp = *(((T*) &h) + component);
			pointsGradient[getPointsIndex(i, 0, 3, pointsSize)] += (fx * fy * hComp + (1 - fx) * fy * gComp + fx * (1 - fy) * fComp + (1 - fx) * (1 - fy) * eComp - fx * fy * dComp - (1 - fx) * fy * cComp - fx * (1 - fy) * bComp - (1 - fx) * (1 - fy) * aComp) * outputGrad; // z
			pointsGradient[getPointsIndex(i, 1, 3, pointsSize)] += (fx * fz * hComp + (1 - fx) * fz * gComp - fx * fz * fComp - (1 - fx) * fz * eComp + fx * (1 - fz) * dComp + (1 - fx) * (1 - fz) * cComp - fx * (1 - fz) * bComp - (1 - fx) * (1 - fz) * aComp) * outputGrad; // y
			pointsGradient[getPointsIndex(i, 2, 3, pointsSize)] += (fy * fz * hComp - fy * fz * gComp + (1 - fy) * fz * fComp - (1 - fy) * fz * eComp + fy * (1 - fz) * dComp - fy * (1 - fz) * cComp + (1 - fy) * (1 - fz) * bComp - (1 - fy) * (1 - fz) * aComp) * outputGrad; // x

		}
	}
}


// Select kernel according to spatial rank and number of components
template<typename T>
void runResampleGradientTextureMemoryKernel(
	const int dims,
	const unsigned int batch,
	const unsigned int dataBatchSize,
	const unsigned int xSize,
	const unsigned int ySize,
	const unsigned int zSize,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int elementsPerKernelCall,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	cudaTextureObject_t dataTexture,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
) {
    int blockSize;
    int minGridSize;
    int gridSize;
	if(dims == 1) {
		if (components == 1) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient1DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient1DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient1DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient1DCudaKernel<float, float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient1DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		}
	} else if (dims == 2) {
		if (components == 1){
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient2DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient2DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient2DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient2DCudaKernel<float, float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient2DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		}
	} else {
		if (components == 1) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient3DCudaKernel<float, float><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 2) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float2>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient3DCudaKernel<float, float2><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else if (components == 3) {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float3>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient3DCudaKernel<float, float3><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		} else {
		    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradient3DCudaKernel<float, float4>, 0, 0);
		    gridSize = (elementsPerKernelCall / components + blockSize - 1) / blockSize;
			ResampleGradient3DCudaKernel<float, float4><<<gridSize, blockSize>>>(batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
		}
	}
}


// Prepare and run texture memory kernel
template <typename T>
void ResampleGradientTextureMemory (
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const T* __restrict__ outputGradient,
	const T* __restrict__ data,
	const T* __restrict__ points,
	T* __restrict__ dataGradient,
	T* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
){
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
			runResampleGradientTextureMemoryKernel(dims, batch, dataBatchSize, xSize, ySize, zSize, components, pointsSize, elementsPerKernelCall, outputSize, outputGradientSize, outputGradient, dataTexture, points, dataGradient, pointsGradient, boundaries);
			HANDLE_ERROR(cudaDeviceSynchronize());
		}
		// Destroy texture object and free memory
		HANDLE_ERROR(cudaDestroySurfaceObject(surfaceObject));
		cudaDestroyTextureObject(dataTexture);
		HANDLE_ERROR(cudaFreeArray(cuArray));
}


// Define the GPU implementation that launches the CUDA kernel.
void LaunchResampleGradientKernel(
	const unsigned int dataBatchSize,
	const int dims,
	const unsigned int* __restrict__ dimSizes,
	const unsigned int components,
	const unsigned int pointsSize,
	const unsigned int outputElementsPerBatch,
	const unsigned int outputSize,
	const unsigned int outputGradientSize,
	const float* __restrict__ outputGradient,
	const float* __restrict__ data,
	const float* __restrict__ points,
	float* __restrict__ dataGradient,
	float* __restrict__ pointsGradient,
	const Boundary* __restrict__ boundaries
) {

	unsigned int dataSize = dataBatchSize * components;
	for (int dim = 0; dim < dims; dim++){
		dataSize *= dimSizes[dim];
	}
	cudaMemset(dataGradient, 0, dataSize * sizeof(float));
	cudaMemset(pointsGradient, 0, pointsSize * sizeof(float));

	// Run kernel with texture memory
	if (dims <= 3 && components <= 4) {
        if((dims == 1 && dimSizes[0] <= 8192)||
           (dims == 2 && dimSizes[0] <= 32768 && dimSizes[1] <= 65536)||
           (dims == 3 && dimSizes[0] <= 2048 && dimSizes[1] <= 2048 && dimSizes[2] <= 2048))
        {
            ResampleGradientTextureMemory<float>(
                dataBatchSize,
                dims,
                dimSizes,
                components,
                pointsSize,
                outputElementsPerBatch,
                outputSize,
                outputGradientSize,
                outputGradient,
                data,
                points,
                dataGradient,
                pointsGradient,
                boundaries
            );
            return;
        }
	}

	// Launch the cuda kernel.
	int blockSize;
	int minGridSize;
	int gridSize;
	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, ResampleGradientCudaKernel<float>, 0, 0);
	gridSize = (outputSize / components + blockSize - 1) / blockSize;

	unsigned int* dimSizesDevice;
	cudaMalloc(&dimSizesDevice, dims * sizeof(unsigned int));
	cudaMemcpy(dimSizesDevice, dimSizes, dims * sizeof(unsigned int), cudaMemcpyHostToDevice);

	float* q;
    cudaMalloc(&q, gridSize * blockSize * dims * sizeof(float));

    float* weights;
    cudaMalloc(&weights, gridSize * blockSize * (dims + 1) * sizeof(float));

	ResampleGradientCudaKernel<float><<<gridSize, blockSize>>>(
		dataBatchSize,
		dims,
		dimSizesDevice,
		components,
		pointsSize,
		outputElementsPerBatch,
		outputSize,
		outputGradientSize,
		outputGradient,
		data,
		points,
		dataGradient,
		pointsGradient,
		boundaries,
		q,
		weights
	);

	HANDLE_ERROR(cudaDeviceSynchronize());
	HANDLE_ERROR(cudaFree(dimSizesDevice));
	HANDLE_ERROR(cudaFree(q));
	HANDLE_ERROR(cudaFree(weights));
}

}  // end namespace tensorflow