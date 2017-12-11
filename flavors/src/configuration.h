#pragma once

#include <ostream>
#include <vector>

#include "utils.h"

namespace Flavors
{
	class Configuration
	{
	public:
		static Configuration Default32;
		static Configuration Binary32;

		static Configuration Default(unsigned depth);
		static Configuration Binary(unsigned depth);

		int Length;
		CudaArray<unsigned> Levels;

		Configuration();
		explicit Configuration(const std::vector<unsigned>& levels);
		void Create(const std::vector<unsigned>& levels);

		unsigned operator[](int level) const;

		int Depth() const { return h_levels.size(); };
		int Mask(int level) const;

		void PopLastLevel();

		std::string ToString();
		friend std::ostream& operator<<(std::ostream& os, Configuration& obj);

		friend bool operator==(const Configuration& lhs, const Configuration& rhs);
		friend bool operator!=(const Configuration& lhs, const Configuration& rhs);

		Configuration(const Configuration& other) = default;
		Configuration(Configuration&& other) noexcept = default;
		Configuration& operator=(const Configuration& other) = default;
		Configuration& operator=(Configuration&& other) noexcept = default;
	private:
		std::vector<unsigned> h_levels;
	};
}
