#ifndef KERNEL_RESAMPLE_
#define KERNEL_RESAMPLE_

#include "helpers.h"

namespace tensorflow {

template<typename Device, typename T>
struct ResampleFunctor {
	void operator()(
		const Device &d,
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
	);
};

}  // namespace tensorflow

#endif //KERNEL_RESAMPLE_
