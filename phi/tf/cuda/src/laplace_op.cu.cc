
#include <cuda_runtime.h>
#include <iostream>


static void CheckCudaErrorAux(const char* file, unsigned line, const char* statement, cudaError_t err) {
  if (err == cudaSuccess) return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;

  exit(1);
}
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)

// Converts coordinates of the simulation grid to indices of the extended mask grid with a shift to get the indices of the neighbors
__device__ int gridIDXWithOffsetShifted(const int *dimensions, const int dim_size, const int *cords, int cords_offset, int dim_index_offset, int offset)
{
    int factor = 1;
    int result = 0;
    for (int i = 0; i < dim_size; i++)
    {
        if (i == dim_index_offset)
            result += factor * (cords[i + cords_offset * dim_size] + offset);
        else
            result += factor * (cords[i + cords_offset * dim_size] + 1);

        factor *= dimensions[i];
    }
    return result;
}

__device__ void CordsByRow(int row, const int *dimensions, const int dim_size, const int dim_product, int *cords)
{
    int modulo = 0;
    int divisor = dim_product;

    for (int i = dim_size - 1; i >= 0; i--)
    {
        divisor = divisor / dimensions[i];

        cords[i + row * dim_size] = (modulo == 0 ? row : (row % modulo)) / divisor; // 0 mod 0 not possible due to c++ restrictions
        modulo = divisor;
    }
}


__global__ void calcLaplaceMatrix(const int *dimensions, const int dim_size, const int dim_product, const float *active_mask, const float *fluid_mask, const int *mask_dimensions, signed char *laplace_matrix, int *cords)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < dim_product; row += blockDim.x * gridDim.x)
    { // TODO: reduce this by half since the matrix is symmetrical and only the half needs to be created? => Already pretty fast
        // Derive the coordinates of the dim_size-Dimensional mask by the laplace row id
        CordsByRow(row, dimensions, dim_size, dim_product, cords);

        // Every thread accesses the laplaceDataBuffer at different areas. index_pointer points to the current position of the current thread
        int index_pointer = row * (dim_size * 2 + 1);

        // forward declaration of variables, that are reused
        int mask_idx = 0;
        int mask_idx_before = 0;
        int mask_idx_after = 0;

		// dim_size-Dimensional "Cubes" have exactly dim_size * 2  "neighbors"
		int diagonal = -dim_size * 2;

        // get the index on the extended mask grid of the current cell
		int rowMaskIdx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, 0, 1);

        // Check neighbors if they are solids. For every solid neighbor increment diagonal by one
		for (int j = dim_size - 1; j >= 0; j--)
        {
            // get the index on the extended mask grid of the neighbor cells
            mask_idx_before = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);
            mask_idx_after = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 2);
            if(active_mask[mask_idx_before] == 0.0f && fluid_mask[mask_idx_before] == 0.0f) diagonal++;
            if(active_mask[mask_idx_after] == 0.0f && fluid_mask[mask_idx_after] == 0.0f) diagonal++;
        }

        // Check the "left"/"before" neighbors if they are fluid and add them to the laplaceData
        for (int j = dim_size - 1; j >= 0; j--)
        {
            mask_idx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 0);

            if (active_mask[mask_idx] == 1 && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            { // fluid - fluid
				laplace_matrix[index_pointer] = 1;
            }
            else if (active_mask[mask_idx] == 0 && fluid_mask[mask_idx] == 1)
            { // Empty / open cell
                // pass, because we initialized the data with zeros
            }
            index_pointer++;
        }

        // Add the diagonal value
		laplace_matrix[index_pointer] = diagonal;
        index_pointer++;

        // Finally add the "right"/"after" neighbors
        for (int j = 0; j < dim_size; j++)
        {
            mask_idx = gridIDXWithOffsetShifted(mask_dimensions, dim_size, cords, row, j, 2);

            if (active_mask[mask_idx] == 1 && fluid_mask[mask_idx] == 1 && !(active_mask[rowMaskIdx] == 0 && fluid_mask[rowMaskIdx] == 0))
            { // fluid - fluid
				laplace_matrix[index_pointer] = 1;
            }
            else if (active_mask[mask_idx] == 0 && fluid_mask[mask_idx] == 1)
            { // Empty / open cell
                // pass, because we initialized the data with zeros
            }
            index_pointer++;
        }
    }
}

__global__ void setUpData( const int dim_size, const int dim_product, signed char *laplace_matrix) {
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < dim_product * (dim_size * 2 + 1);
         row += blockDim.x * gridDim.x)
    {
        laplace_matrix[row] = 0;
    }
}
void LaplaceMatrixKernelLauncher(const int *dimensions, const int dim_size, const int dim_product,  const float *active_mask, const float *fluid_mask, const int *mask_dimensions, signed char *laplace_matrix, int *cords) {
    // get block and gridSize to theoretically get best occupancy
    int blockSize;
    int minGridSize;
    int gridSize;
    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              setUpData, 0, 0);
    gridSize = (dim_product * (dim_size * 2 + 1) + blockSize - 1) / blockSize;

    // Init Laplace Matrix with zeros
    setUpData<<<gridSize, blockSize>>>(dim_size, dim_product, laplace_matrix);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());


    cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,
                              calcLaplaceMatrix, 0, 0);
    gridSize = (dim_product + blockSize - 1) / blockSize;

    // Calculate the Laplace Matrix
    calcLaplaceMatrix<<<gridSize, blockSize>>>(dimensions, dim_size, dim_product, active_mask, fluid_mask, mask_dimensions, laplace_matrix, cords);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
}

