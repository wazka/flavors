#include "configuration.h"

#include <algorithm>
#include <thrust/execution_policy.h>

namespace Flavors
{
	Configuration Configuration::Default32{ std::vector <unsigned>{32} };
	Configuration Configuration::Binary32{ std::vector <unsigned>{1, 1, 1, 1, 1, 1, 1, 1,
																  1, 1, 1, 1, 1, 1, 1, 1,
																  1, 1, 1, 1, 1, 1, 1, 1,
																  1, 1, 1, 1, 1, 1, 1, 1} };

	Configuration::Configuration():
		Length(0)
	{
	}

	Configuration::Configuration(const std::vector<unsigned>& levels):
		Length(0),
		h_levels(levels),
		Levels(levels.size())
	{
		for (auto level : h_levels)
			Length += level;

		cuda::memory::copy(Levels.Get(), h_levels.data(), Depth() * sizeof(unsigned));
	}

	unsigned Configuration::operator[](int level) const
	{
		return h_levels[level];
	}

	int Configuration::Mask(int level) const
	{
		//TODO: Can this be done better? Why overflow acts differently for literal 32 and config.h_levels[level]
		return h_levels[level] != 32 ? (1u << h_levels[level]) - 1u : ~0;
	}

	void Configuration::PopLastLevel()
	{
		h_levels.pop_back();
	}

	std::ostream& operator<<(std::ostream& os, const Configuration& obj)
	{
		os << "{";
		std::copy(obj.h_levels.begin(), obj.h_levels.end() - 1, std::ostream_iterator<unsigned>(os, ", "));
		os << *(obj.h_levels.end() - 1) << "}";

		return os;
	}

	bool operator==(const Configuration& lhs, const Configuration& rhs)
	{
		if (lhs.Depth() != rhs.Depth())
			return false;

		auto cmpResult = std::mismatch(lhs.h_levels.begin(), lhs.h_levels.end(), rhs.h_levels.begin());
		return cmpResult.first == lhs.h_levels.end();
	}

	bool operator!=(const Configuration& lhs, const Configuration& rhs)
	{
		return !(lhs == rhs);
	}
}
