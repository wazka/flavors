#pragma once
#include "keys.h"
#include <ostream>

namespace Flavors
{
	class Masks : public Keys
	{
	public:
		Masks(const Configuration& config, int count);
		Masks(const Configuration& config, int count, unsigned* data, unsigned* lengths);

		CudaArray<unsigned> Lengths;

		void FillRandom(int seed) override;
		void FillRandom(int seed, int max, int min = 1);

		void Sort() override;

		friend std::ostream& operator<<(std::ostream& os, const Masks& obj);

		Masks ReshapeMasks(Configuration& newConfig);

		friend bool operator==(const Masks& lhs, const Masks& rhs);

		friend bool operator!=(const Masks& lhs, const Masks& rhs);

		virtual ~Masks() = default;

		Masks(const Masks& other) = delete;
		Masks(Masks&& other) noexcept = default;
		Masks& operator=(const Masks& other) = delete;
		Masks& operator=(Masks&& other) = default;
	};
}
