#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <sys/time.h>

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("PressureSolve")
    .Input("dimensions: int32")

    .Input("mask_dimensions: int32")
    .Input("active_mask: float32")
    .Input("fluid_mask: float32")
    .Input("laplace_matrix: int8")

    .Input("divergence: float32")
    .Input("p: float32")
    .Input("r: float32")
    .Input("z: float32")
    .Input("pressure: float32")
    .Input("one_vector: float32")

    .Attr("dim_product: int")
    .Attr("accuracy: float")
    .Attr("max_iterations: int")
    .Output("pressure_out: float32")
    .Output("iterations: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(5)); // divergence
        return Status::OK();
    });


void LaunchPressureKernel(const int *dimensions, const int dimProduct, const int dimSize,
                          const signed char* laplaceMatrix,
                          float* p, float* z, float* r, float *divergence, float* x,
                          const float* oneVector,
                          bool* thresholdReached,
                          const float accuracy,
                          const int max_iterations,
                          const int batch_size,
                          int* iterations_gpu);

void LaplaceMatrixKernelLauncher(const int *dimensions, const int dimSize, const int dimProduct, const float *active_mask, const float *fluid_mask, const int *maskDimensions, signed char *laplaceMatrix, int *cords);

class PressureSolveOp : public OpKernel {
    private:
        int dim_product;
        float accuracy;
        int max_iterations;

    public:
        explicit PressureSolveOp(OpKernelConstruction* context) : OpKernel(context) {
        context->GetAttr("dim_product", &dim_product);
        context->GetAttr("accuracy", &accuracy);
        context->GetAttr("max_iterations", &max_iterations);
    }

    void Compute(OpKernelContext* context) override {
        auto begin = std::chrono::high_resolution_clock::now();

        // General
        const Tensor& input_dimensions = context->input(0);

        // Laplace related
        const Tensor &input_mask_dimensions = context->input(1);
        const Tensor &input_active_mask = context->input(2);
        const Tensor &input_fluid_mask = context->input(3);
        Tensor input_laplace_matrix = context->input(4);

        // Pressure Solve
        Tensor input_divergence = context->input(5);
        Tensor input_p = context->input(6);
        Tensor input_r = context->input(7);
        Tensor input_z = context->input(8);
        Tensor input_pressure = context->input(9);
        const Tensor& input_one_vector = context->input(10);

        // Flattening
        auto dimensions = input_dimensions.flat<int32>();

        auto mask_dimensions = input_mask_dimensions.flat<int32>();
        auto active_mask = input_active_mask.flat<float>();
        auto fluid_mask = input_fluid_mask.flat<float>();
        auto laplace_matrix = input_laplace_matrix.flat<int8>();

        auto divergence = input_divergence.flat<float>();
        auto p = input_p.flat<float>();
        auto r = input_r.flat<float>();
        auto z = input_z.flat<float>();
        auto pressure = input_pressure.flat<float>();
        auto one_vector = input_one_vector.flat<float>();

        int batch_size = input_divergence.shape().dim_size(0);
        int dim_size = dimensions.size();

        auto end = std::chrono::high_resolution_clock::now();

//        printf("General Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

        begin = std::chrono::high_resolution_clock::now();
        // Laplace:
        // Laplace Helper
        Tensor cords; // cords allocation does not really impact the performance. However it could be outsourced to be reused.
        TensorShape cords_shape;
        cords_shape.AddDim(dim_product);
        cords_shape.AddDim(dim_size);
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_INT32, cords_shape, &cords));
        auto cords_flat = cords.flat<int32>();

        end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


        begin = std::chrono::high_resolution_clock::now();
        LaplaceMatrixKernelLauncher(dimensions.data(), dim_size, dim_product, active_mask.data(), fluid_mask.data(), mask_dimensions.data(), laplace_matrix.data(), cords_flat.data());
        end = std::chrono::high_resolution_clock::now();

//        printf("Laplace Matrix Generation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


        begin = std::chrono::high_resolution_clock::now();

        TensorShape threshold_shape;
        threshold_shape.AddDim(batch_size);
        Tensor threshold_reached_tensor;
        OP_REQUIRES_OK(context, context->allocate_temp(DataType::DT_BOOL, threshold_shape, &threshold_reached_tensor));
        auto threshold_reached = threshold_reached_tensor.flat<bool>();

        context->set_output(0, input_pressure);

        TensorShape iterations_shape;
        iterations_shape.AddDim(1);
        Tensor* iterations_tensor;

        OP_REQUIRES_OK(context, context->allocate_output(1, iterations_shape, &iterations_tensor));
        auto iterations_flat = iterations_tensor->flat<int>();

        end = std::chrono::high_resolution_clock::now();


//        printf("Pressure Solve Preparation took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);


        begin = std::chrono::high_resolution_clock::now();
        LaunchPressureKernel(dimensions.data(), dim_product, dim_size,
                              laplace_matrix.data(),
                              p.data(), z.data(), r.data(), divergence.data(), pressure.data(),
                              one_vector.data(),
                              threshold_reached.data(),
                              accuracy,
                              max_iterations,
                              batch_size,
                              iterations_flat.data());
        end = std::chrono::high_resolution_clock::now();


//        printf("Pressure Solve took: %f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);
//        printf("%f\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() * 1e-6);

  }
};

REGISTER_KERNEL_BUILDER(Name("PressureSolve").Device(DEVICE_GPU), PressureSolveOp);
