#include "helpers.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

// Op registration
namespace tensorflow {
REGISTER_OP("Resample")
	.Attr("T: {bfloat16, float, double}")
	.Input("data: T")
	.Input("points: T")
	.Input("boundaries: uint32")
	.Output("out: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		::tensorflow::shape_inference::ShapeHandle dataShape = c->input(0);
		::tensorflow::shape_inference::ShapeHandle pointsShape = c->input(1);
		::tensorflow::shape_inference::ShapeHandle outputShape;
		::tensorflow::shape_inference::DimensionHandle batchSize;
		TF_RETURN_IF_ERROR(c->Max(c->Dim(dataShape, 0), c->Dim(pointsShape, 0), &batchSize));
		TF_RETURN_IF_ERROR(c->ReplaceDim(outputShape, 0, batchSize, &outputShape));
		TF_RETURN_IF_ERROR(c->ReplaceDim(pointsShape, c->Rank(pointsShape) - 1, c->Dim(dataShape, c->Rank(dataShape) - 1), &outputShape));
		c->set_output(0, outputShape);
		return Status::OK();
	});


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
);


// OpKernel definition.
class ResampleOp: public OpKernel {
public:
	explicit ResampleOp(OpKernelConstruction *context) : OpKernel(context) {}

	void Compute(OpKernelContext *context) override {
		// Grab the input tensors
		const Tensor &data = context->input(0);
		const Tensor &points = context->input(1);
		const Tensor &boundaries = context->input(2);

		// Prepare data access parameters
		assert(data.shape().dims() >= 2);
		// dataBatchSize
		const unsigned int dataBatchSize = data.shape().dim_size(0);
		const unsigned int pointsBatchSize = points.shape().dim_size(0);
		assert(dataBatchSize == pointsBatchSize || dataBatchSize == 1 || pointsBatchSize == 1);
		const unsigned int outputBatchSize = dataBatchSize > pointsBatchSize ? dataBatchSize : pointsBatchSize;
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

		// Create output shape
		TensorShape outputShape = points.shape();
		outputShape.set_dim(0, outputBatchSize);
		outputShape.set_dim(outputShape.dims() - 1, components);


		// Create an output tensor
		Tensor *output = NULL;
		OP_REQUIRES_OK(
			context,
			context->allocate_output(0, outputShape, &output)
		);

		// outputElementsPerBatch
		const unsigned int outputElementsPerBatch = output->NumElements() / outputBatchSize;

		// Do the computation.
		LaunchResampleKernel(
			dataBatchSize,
			dims,
			dimSizes,
			components,
			pointsSize,
			outputElementsPerBatch,
			output->NumElements(),
			data.flat<float>().data(),
			points.flat<float>().data(),
			output->flat<float>().data(),
			(Boundary*) boundaries.flat<unsigned int>().data()
		);
	}
};


// Register the GPU kernel.
REGISTER_KERNEL_BUILDER(Name("Resample").Device(DEVICE_GPU), ResampleOp);

} // namespace tensorflow
