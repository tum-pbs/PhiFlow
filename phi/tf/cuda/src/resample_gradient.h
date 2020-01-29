#ifndef RESAMPLE_GRADIENT_H_
#define RESAMPLE_GRADIENT_H_

#include "helpers.h"

namespace tensorflow {

template<typename Device, typename T>
struct ResampleGradientFunctor {
	void operator()(
		const Device &d,
		const unsigned int dataBatchSize,
		const int dims,
		const unsigned int* __restrict__ dimSizes,
		const unsigned int components,
		const unsigned int pointsSize,
		const unsigned int outputElementsPerBatch,
		const unsigned int outputSize,
		const T* __restrict__ outputGradient,
		const T* __restrict__ data,
		const T* __restrict__ points,
		T* __restrict__ dataGradient,
		T* __restrict__ pointsGradient,
		const Boundary* __restrict__ boundaries
	);
};

}  // namespace tensorflow



#endif /* RESAMPLE_GRADIENT_H_ */
