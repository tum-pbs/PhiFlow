#include "helpers.h"
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
);


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
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
		LaunchResampleGradientKernel(
			dataBatchSize,
			dims,
			dimSizes,
			components,
			pointsSize,
			outputElementsPerBatch,
			outputSize,
			outputGradient.NumElements(),
			outputGradient.flat<float>().data(),
			data.flat<float>().data(),
			points.flat<float>().data(),
			dataGradient->flat<float>().data(),
			pointsGradient->flat<float>().data(),
			(Boundary*) boundaries.flat<unsigned int>().data()
		);
	}
};


// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(Name("ResampleGradient").Device(DEVICE_GPU), ResampleGradientOp);

} // namespace tensorflow
