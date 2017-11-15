#pragma once
#include "utils.h"
#include "configuration.h"
#include <ostream>

namespace Flavors
{
	class Keys
	{
	public:
		Keys(const Configuration& config, int count);
		Keys(const Configuration& config, int count, unsigned* data);

		virtual void FillRandom(int seed);
		std::vector<std::vector<unsigned>> ToHost() const;

		friend std::ostream& operator<<(std::ostream& os, const Keys& obj);

		int Depth() const { return Store.Depth; }

		Keys ReshapeKeys(Configuration& newConfig);

		virtual void Sort();
		bool Sorted() { return Permutation.Get() != nullptr; };

		friend bool operator==(const Keys& lhs, const Keys& rhs);
		friend bool operator!=(const Keys& lhs, const Keys& rhs);

		Cuda2DArray Store;
		Configuration Config;
		int Count;
		CudaArray<unsigned> Permutation;

		Cuda2DArray Borders();

		virtual ~Keys() = default;

		Keys(const Keys& other) = delete;
		Keys(Keys&& other) noexcept = default;
		Keys& operator=(const Keys& other) = delete;
		Keys& operator=(Keys&& other) = default;

	protected:
		void launchReshape(Configuration& newConfig, Keys& newKeys);
		void copyPermutation(Keys& newKeys);
		void initPermutation();
		void updatePermutation(int level, CudaArray<unsigned>& tmp);
		void applyPermutation(int level, CudaArray<unsigned>& tmp);
	};
}
