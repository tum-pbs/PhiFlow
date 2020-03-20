#ifndef HELPERS_H_
#define HELPERS_H_

// Declare functions as host device for NVCC
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

// Function declarations - parsed by NVCC and GCC
CUDA_HOSTDEV
bool checkBit(int var, int pos);

template<typename T>
CUDA_HOSTDEV
unsigned int getDataIndex(const unsigned int batch, const T* q, const unsigned int component, const int dims, const unsigned int* dimSizes, const unsigned int components);

CUDA_HOSTDEV
unsigned int getPointsIndex(const unsigned int i, int dim, int dims, const unsigned int pointsSize);

CUDA_HOSTDEV
int pow2(int exp);

enum Boundary : unsigned int;

template<typename T>
CUDA_HOSTDEV
inline T mod(T k, T n);

template<typename T>
CUDA_HOSTDEV
bool applyBoundaries(const Boundary* boundaries, T* q, const int dims, const unsigned int* dimSizes);

template<typename T>
T fetchDataHost(const T* data, const Boundary* boundaries, const unsigned int batch, T* q, const unsigned int component, const int dims, const unsigned int* dimSizes, const unsigned int components);


// Function definitions - parsed only by NVCC
#ifdef __CUDACC__

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

namespace tensorflow {

} // Namespace tensorflow

// https://stackoverflow.com/questions/13245258/handle-error-not-found-error-in-cuda/13245319
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

__host__ __device__
bool checkBit(int var, int pos){
	return var & (1 << pos);
}


// get index in multidimensional data array
template<>
__host__ __device__
unsigned int getDataIndex(const unsigned int batch, const unsigned int* q, const unsigned int component, const int dims, const unsigned int* dimSizes, const unsigned int components) {
	unsigned int index = component;
	unsigned int multiplier = components;
	for(int dim = dims - 1; dim >= 0; dim--){
		index += q[dim] * multiplier;
		multiplier *= dimSizes[dim];
	}
	index += batch * multiplier;
	return index;
}


template<typename T>
__host__ __device__
unsigned int getDataIndex(const unsigned int batch, const T* q, const unsigned int component, const int dims, const unsigned int* dimSizes, const unsigned int components) {
	unsigned int index = component;
	unsigned int multiplier = components;
	for(int dim = dims - 1; dim >= 0; dim--){
		index += ((unsigned int) round(q[dim])) * multiplier;
		multiplier *= dimSizes[dim];
	}
	index += batch * multiplier;
	return index;
}

template
__host__ __device__
unsigned int getDataIndex(const unsigned int, const float*, const unsigned int, const int, const unsigned int*, const unsigned int);

enum Boundary : unsigned int {ZERO, REPLICATE, CIRCULAR, SYMMETRIC, REFLECT};

// Float modulo with positive result
template<typename T>
__host__ __device__
inline T mod(T k, T n) {
	k = fmod(k, n);
	return k < 0 ? k + n : k;
}


// Change point q according to boundary condition
// Returns false if q is outside the field and boundary condition is ZERO
template<typename T>
__host__ __device__
bool applyBoundaries(const Boundary* boundaries, T* q, const int dims, const unsigned int* dimSizes) {
	for (int dim = 0; dim < dims; dim++) {
		T qDim = q[dim];
		const unsigned int dimSize = dimSizes[dim];
		if (qDim < 0) {
			Boundary lowerBoundary = boundaries[2 * dim];
			switch(lowerBoundary) {
				case ZERO:
					return false;
				case REPLICATE:
					q[dim] = 0;
					break;
				case CIRCULAR:
					q[dim] = mod(qDim, (T) dimSize);
					break;
				case SYMMETRIC:
					qDim = mod((-qDim - 1), ((T) (2 * dimSize)));
					if (qDim > dimSize - 1) {
						qDim = 2 * dimSize - qDim - 1;
					}
					q[dim] = qDim;
					break;
				case REFLECT:
				    qDim = mod((-qDim), ((T) (2 * dimSize - 2)));
					if (qDim > dimSize - 1) {
						qDim = 2 * dimSize - qDim - 2;
					}
					q[dim] = qDim;
					break;
			}
		} else if (qDim > dimSize - 1) {
			Boundary upperBoundary = boundaries[2 * dim + 1];
			switch(upperBoundary) {
				case ZERO:
					return false;
				case REPLICATE:
					q[dim] = dimSize - 1;
					break;
				case CIRCULAR:
					q[dim] = fmod(qDim, (T) dimSize);
					break;
				case SYMMETRIC:
					qDim = fmod(qDim, ((T) (2 * dimSize)));
					if (qDim > dimSize - 1) {
						qDim = 2 * dimSize - qDim - 1;
					}
					q[dim] = qDim;
					break;
				case REFLECT:
				    qDim = fmod(qDim, ((T) (2 * dimSize - 2)));
					if (qDim > dimSize - 1) {
						qDim = 2 * dimSize - qDim - 2;
					}
					q[dim] = qDim;
					break;
			}
		}
	}
	return true;
}

template
__host__ __device__
bool applyBoundaries(const Boundary*, float*, const int, const unsigned int*);

// Retrieve data from device array according to position and boundary condition
template<typename T>
__device__
T fetchDataDevice(const T* data, const Boundary* boundaries, const unsigned int batch, T* q, const unsigned int component, const int dims, const unsigned int* dimSizes, const unsigned int components) {
	if(applyBoundaries(boundaries, q, dims, dimSizes)){
		return ldg(data + getDataIndex(batch, q, component, dims, dimSizes, components));
	} else {
		return 0.0;
	}
}


__host__ __device__
unsigned int getPointsIndex(const unsigned int i, int dim, int dims, const unsigned int pointsSize){
	unsigned int index = i * dims + dim;
	index = index % pointsSize;
	return index;
}


__host__ __device__
int pow2(int exp) {
	int power = 2;
	for (int i = 1; i < exp; i++){
		power *= 2;
	}
	return power;
}


cudaArray* createArray(const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int components) {
	cudaArray* cuArray;
	// Create cuda extent
	cudaExtent extent = make_cudaExtent(xSize, ySize, zSize);

	// Create channel format description
	cudaChannelFormatDesc channelDesc;
	if (components == 1) {
		channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	} else if (components == 2) {
		channelDesc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	} else {
		channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	}

	// Create array
	cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArraySurfaceLoadStore);
	return cuArray;
}


template<typename T>
cudaMemcpy3DParms createCopyParams(const T* __restrict__ data, cudaArray* cuArray, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int components) {
	cudaMemcpy3DParms copyParams = {0};
	cudaExtent extent = make_cudaExtent(xSize, ySize, zSize);
	copyParams.srcPtr = make_cudaPitchedPtr((void*) data, xSize * components * sizeof(T), xSize, ySize);
	copyParams.dstArray = cuArray;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	return copyParams;
}


cudaResourceDesc createResDesc(cudaArray* cuArray) {
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	return resDesc;
}


cudaTextureObject_t createTextureObject(cudaArray* cuArray) {
	cudaTextureObject_t dataTexture = 0;
	// Specify texture
	cudaResourceDesc resDesc = createResDesc(cuArray);

	// Specify texture object
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModePoint;
	//texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	//Create texture object
	cudaCreateTextureObject(&dataTexture, &resDesc, &texDesc, NULL);
	return dataTexture;
}


cudaSurfaceObject_t createSurfaceObject(cudaArray* cuArray){
	cudaSurfaceObject_t surfaceObject = 0;
	cudaResourceDesc resDesc = createResDesc(cuArray);
	cudaCreateSurfaceObject(&surfaceObject, &resDesc);
	return surfaceObject;
}

// Copy float3 data to float4 texture
__global__
void CopyKernel (const float* data, cudaSurfaceObject_t surfaceObject, int dims, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int batch) {
	unsigned int dataElementsPerBatch = xSize * ySize * zSize;
	for (unsigned int i = batch * dataElementsPerBatch + blockIdx.x * blockDim.x + threadIdx.x; i < batch * dataElementsPerBatch + dataElementsPerBatch; i += blockDim.x * gridDim.x) {
	    unsigned int index = i % dataElementsPerBatch;
		unsigned int x = index % xSize;
		unsigned int y = (index / xSize) % ySize;
		unsigned int z = index / (xSize * ySize);
		float4 element;
		element.x = ldg(data + 3 * i);
		element.y = ldg(data + 3 * i + 1);
		element.z = ldg(data + 3 * i + 2);
		element.w = 0;
		if (dims == 1) {
			surf1Dwrite(element, surfaceObject, x * sizeof(float4));
		} else if (dims == 2) {
			surf2Dwrite(element, surfaceObject, x * sizeof(float4), y);
		} else {
			surf3Dwrite(element, surfaceObject, x * sizeof(float4), y, z);
		}
	}
}

// Copy data to texture array according to spatial rank and numer of components
template<typename T>
void copyDataToArray(const T* __restrict__ data, cudaArray* cuArray, cudaSurfaceObject_t surfaceObject, cudaMemcpy3DParms copyParams, const int dims, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize, const unsigned int batch, const unsigned int components) {
	if (components == 3) {
		// Use Surface to write to texture array
		int blockSize;
		int minGridSize;
		int gridSize;
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, CopyKernel, 0, 0);
		gridSize = (xSize * ySize * zSize + blockSize - 1) / blockSize;
		CopyKernel<<<gridSize, blockSize>>>(data, surfaceObject, dims, xSize, ySize, zSize, batch);
		HANDLE_ERROR(cudaDeviceSynchronize());
	} else if (dims <= 2) {
		cudaMemcpyToArray(cuArray, 0, 0, data + batch * xSize * ySize * zSize * components, xSize * ySize * zSize * components * sizeof(T), cudaMemcpyDeviceToDevice);
	} else {
		copyParams.srcPtr = make_cudaPitchedPtr((void*) (data + batch * xSize * ySize * zSize * components), xSize * components * sizeof(T), xSize, ySize);
		cudaMemcpy3D(&copyParams);
	}
}

// Funtion template for retrieving data from texture memory
template<typename T>
__device__
inline T tex1DHelper(cudaTextureObject_t texObj, float x) {
	return tex1D<T>(texObj, x);
}

// Specialised definition for retrieving float3 from float4 texture
template<>
__device__
inline float3 tex1DHelper(cudaTextureObject_t texObj, float x) {
	float4 texel = tex1D<float4>(texObj, x);
	float3 result;
	result.x = texel.x;
	result.y = texel.y;
	result.z = texel.z;
	return result;
}

// Retrieve data from texture memory according to position and boundary condition
template<typename T>
__device__
inline T tex1DHelper(cudaTextureObject_t texObj, float x, const Boundary* boundaries, const unsigned int xSize) {
	if (applyBoundaries(boundaries, &x, 1, &xSize)) {
		return tex1DHelper<T>(texObj, x + 0.5);
	} else {
		T v;
		memset(&v, 0, sizeof(T));
		return v;
	}
}


template<typename T>
__device__
inline T tex2DHelper(cudaTextureObject_t texObj, float x, float y) {
	return tex2D<T>(texObj, x, y);
}


template<>
__device__
inline float3 tex2DHelper(cudaTextureObject_t texObj, float x, float y) {
	float4 texel = tex2D<float4>(texObj, x, y);
	float3 result;
	result.x = texel.x;
	result.y = texel.y;
	result.z = texel.z;
	return result;
}


template<typename T>
__device__
inline T tex2DHelper(cudaTextureObject_t texObj, float x, float y, const Boundary* boundaries, const unsigned int xSize, const unsigned int ySize) {
	float q[2] = {y, x};
	unsigned int dimSizes[2] = {ySize, xSize};
	if (applyBoundaries(boundaries, q, 2, dimSizes)) {
		return tex2DHelper<T>(texObj, q[1] + 0.5, q[0] + 0.5);
	} else {
		T v;
		memset(&v, 0, sizeof(T));
		return v;
	}
}


template<typename T>
__device__
inline T tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z) {
	return tex3D<T>(texObj, x, y, z);
}


template<>
__device__
inline float3 tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z) {
	float4 texel = tex3D<float4>(texObj, x, y, z);
	float3 result;
	result.x = texel.x;
	result.y = texel.y;
	result.z = texel.z;
	return result;
}


template<typename T>
__device__
inline T tex3DHelper(cudaTextureObject_t texObj, float x, float y, float z, const Boundary* boundaries, const unsigned int xSize, const unsigned int ySize, const unsigned int zSize) {
	float q[3] = {z, y, x};
	unsigned int dimSizes[3] = {zSize, ySize, xSize};
	if (applyBoundaries(boundaries, q, 3, dimSizes)) {
		return tex3DHelper<T>(texObj, q[2] + 0.5, q[1] + 0.5, q[0] + 0.5);
	} else {
		T v;
		memset(&v, 0, sizeof(T));
		return v;
	}
}


#endif // __NVCC__

#endif // HELPERS_H_
