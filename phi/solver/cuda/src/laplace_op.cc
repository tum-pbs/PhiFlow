//nsight -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow; // NOLINT(build/namespaces)

REGISTER_OP("LaplaceMatrix")
    .Input("dimensions: int32")
    .Input("mask_dimensions: int32")
    .Input("active_mask: float32")
    .Input("fluid_mask: float32")
    .Attr("dim_product: int")
    .Input("input_laplace_matrix: int8")
    .Output("laplace_matrix: int8");
    //.SetIsStateful();

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dim_size, const int dimProduct, const float *active_mask, const float *fluid_mask, const int *maskDimensions, signed char *laplaceMatrix, int *cords);

class LaplaceMatrixOp : public OpKernel
{
    private:
        int dim_product;
  
    public:
        explicit LaplaceMatrixOp(OpKernelConstruction *context) : OpKernel(context) {
            context->GetAttr("dim_product", &dim_product);
        }

    void Compute(OpKernelContext *context) override
    {
        // This Op is only required for benchmarking the Laplace Matrix generation speed.
        // The pressure solve Op calls the LaplaceMatrixKernelLauncher before solving the pressure
        const Tensor &input_dimensions = context->input(0);
        const Tensor &input_mask_dimensions = context->input(1);
        const Tensor &input_active_mask = context->input(2);
        const Tensor &input_fluid_mask = context->input(3);
        Tensor input_laplace_matrix = context->input(4);

        auto dimensions = input_dimensions.flat<int32>();
        auto mask_dimensions = input_mask_dimensions.flat<int32>();
        auto active_mask = input_active_mask.flat<float>();
        auto fluid_mask = input_fluid_mask.flat<float>();
        auto laplace_matrix = input_laplace_matrix.flat<int8>();

        int dim_size = dimensions.size();

        Tensor cords;
        TensorShape cords_shape;
        cords_shape.AddDim(dim_product);
        cords_shape.AddDim(input_dimensions.dims());

        context->set_output(0, input_laplace_matrix);

        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32, cords_shape, &cords));

        auto cords_flat = cords.flat<int32>();

        LaplaceMatrixKernelLauncher(dimensions.data(), dim_size, dim_product, active_mask.data(), fluid_mask.data(), mask_dimensions.data(), laplace_matrix.data(), cords_flat.data());
    }
};

REGISTER_KERNEL_BUILDER(Name("LaplaceMatrix").Device(DEVICE_GPU), LaplaceMatrixOp);
