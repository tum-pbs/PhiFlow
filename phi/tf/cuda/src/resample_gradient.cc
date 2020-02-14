#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include "resample_gradient.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("ResampleGradient")
	.Attr("T: {bfloat16, float, double}")
	.Input("output_gradient: T")
	.Input("data: T")
	.Input("points: T")
	.Input("boundaries: uint32")
	.Output("data_gradient: T")
	.Output("points_gradient: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		c->set_output(0, c->input(1));
		c->set_output(1, c->input(2));
		return Status::OK();
	});

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


// CPU specialization of actual computation.
template<typename T>
struct ResampleGradientFunctor<CPUDevice, T> {
	void operator()(
		const CPUDevice &d,
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
	) {
		std::cout << "Gradient CPU" << std::endl;
		unsigned int dataSize = dataBatchSize * components;
		for (int dim = 0; dim < dims; dim++){
			dataSize *= dimSizes[dim];
		}
		memset(dataGradient, 0, dataSize * sizeof(T));
		memset(pointsGradient, 0, pointsSize * sizeof(T));
		for (unsigned int i = 0; i < outputSize / components; i++){
			unsigned int dataBatch = (i * components / outputElementsPerBatch) % dataBatchSize;
			T p[dims];
			for (int dim = 0; dim < dims; dim++){
				p[dim] = points[getPointsIndex(i, dim, dims, pointsSize)];
			}
			int n = pow2(dims);
			for (int j = 0; j < n; j++) {
				T q[dims];
				T weights[dims + 1];
				std::fill(weights, weights + dims + 1, 1);
				for (int dim = 0; dim < dims; dim++){
					T weight = 1;
					if (checkBit(j, dim)) {
						q[dim] = floor(p[dim]) + 1;
						weight = 1 - (q[dim] - p[dim]);
					} else {
						q[dim] = floor(p[dim]);
						weight = 1 - (p[dim] - q[dim]);
					}
					for (int otherDim = 0; otherDim <= dims; otherDim++){
						if(otherDim != dim) {
							weights[otherDim] *= weight;
						}
					}
				}
				for (unsigned int component = 0; component < components; component++) {
					T dataValue = 0;
					T outputGradientValue = outputGradient[(i * components + component) % outputGradientSize];
					if (applyBoundaries(boundaries, q, dims, dimSizes)) {
						unsigned int dataIndex = getDataIndex(dataBatch, q, component, dims, dimSizes, components);
						dataGradient[dataIndex] += weights[dims] * outputGradientValue;
						dataValue = data[dataIndex];
					}
					for (int dim = 0; dim < dims; dim++) {
						int factor = checkBit(j, dim) ? 1 : -1;
						pointsGradient[getPointsIndex(i, dim, dims, pointsSize)] += factor * weights[dim] * dataValue * outputGradientValue;
					}
				}
			}
		}
	}
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template<typename Device, typename T>
class ResampleGradientOp: public OpKernel {
public:
	explicit ResampleGradientOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override {
		// Grab the input tensors
		const Tensor &outputGradient = context->input(0);
		const Tensor &data = context->input(1);
		const Tensor &points = context->input(2);
		const Tensor &boundaries = context->input(3);

		// Prepare data access parameters
		assert(data.shape().dims() >= 2);
		// dataBatchSize
		const unsigned int dataBatchSize = data.shape().dim_size(0);
		const unsigned int pointsBatchSize = points.shape().dim_size(0);
		assert(dataBatchSize == pointsBatchSize || dataBatchSize == 1 || pointsBatchSize == 1);
		//const unsigned int outputBatchSize = outputGradient.shape().dim_size(0);
		unsigned int outputBatchSize = dataBatchSize > pointsBatchSize ? dataBatchSize : pointsBatchSize;
		// dims
		const int dims = data.shape().dims() - 2;
		assert(dims == points.shape().dim_size(points.shape().dims() - 1));
		assert(dims == boundaries.shape().dim_size(0) && boundaries.shape().dim_size(1) == 2);
		// dimSizes
		unsigned int dimSizes[dims];
		for(int i = 0; i < dims; i++){
			dimSizes[i] = data.shape().dim_size(i + 1);
		}
		// components
		const unsigned int components = data.shape().dim_size(data.shape().dims() - 1);
		// pointsSize
		const unsigned int pointsSize = points.NumElements();

		// Create output tensors
		Tensor *dataGradient = NULL;
		Tensor *pointsGradient = NULL;
		OP_REQUIRES_OK(
			context,
			context->allocate_output(0, data.shape(), &dataGradient)
		);
		OP_REQUIRES_OK(
			context,
			context->allocate_output(1, points.shape(), &pointsGradient)
		);

        //outputSize
        unsigned int outputSize = 1;
        for(int i = 1; i < points.shape().dims() - 1; i++) {
            outputSize *= points.shape().dim_size(i);
        }
        outputSize *= outputBatchSize * components;
        //unsigned int outputSize = outputGradient.NumElements() * outputBatchSize;

		// outputElementsPerBatch
		const unsigned int outputElementsPerBatch = outputSize / outputBatchSize;

		// Do the computation.
		/*OP_REQUIRES(
			context,
			data.NumElements() <= tensorflow::kint32max,
			errors::InvalidArgument("Too many elements in tensor.")
		);*/

		ResampleGradientFunctor<Device, T>()(
			context->eigen_device<Device>(),
			dataBatchSize,
			dims,
			dimSizes,
			components,
			pointsSize,
			outputElementsPerBatch,
			outputSize,
			outputGradient.NumElements(),
			outputGradient.flat<T>().data(),
			data.flat<T>().data(),
			points.flat<T>().data(),
			dataGradient->flat<T>().data(),
			pointsGradient->flat<T>().data(),
			(Boundary*) boundaries.flat<unsigned int>().data()
		);
	}
};


// Register the CPU kernels.
#define REGISTER_CPU(T)												\
	REGISTER_KERNEL_BUILDER(										\
		Name("ResampleGradient").Device(DEVICE_CPU).TypeConstraint<T>("T"),	\
		ResampleGradientOp<CPUDevice, T>									\
	);

//REGISTER_CPU(bfloat16);
REGISTER_CPU(float);
//REGISTER_CPU(double);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)												\
	extern template struct ResampleGradientFunctor<GPUDevice, T>;           \
	REGISTER_KERNEL_BUILDER(										\
		Name("ResampleGradient").Device(DEVICE_GPU).TypeConstraint<T>("T"),	\
		ResampleGradientOp<GPUDevice, T>									\
	);

//REGISTER_GPU(bfloat16);
REGISTER_GPU(float);
//REGISTER_GPU(double);

#endif  // GOOGLE_CUDA

} // namespace tensorflow
