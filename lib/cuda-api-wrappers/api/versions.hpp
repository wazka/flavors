/**
 * @file versions.hpp
 *
 * @brief Wrappers for Runtime API functions involving versions -
 * of the CUDA runtime && of the CUDA driver. Also defines a @ref cuda::version_t
 * class for working with such versions (as they are not really single
 * numbers) - which is what the wrappers return.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_VERSIONS_HPP_

#include "error.hpp"
#include <ostream>
#include <utility>

namespace cuda {

using combined_version_t = int;

/**
 * CUDA Runtime version
 *
 * @note not to be confused with @ref device::compute_capability_t !
 */
struct version_t {
	int major;
	int minor;

	static version_t from_single_number(combined_version_t combined_version)
	{
		return { combined_version / 1000, (combined_version % 100) / 10 };
	}

	operator std::pair<int, int>() const { return { major, minor}; }

};

inline std::ostream& operator<<(std::ostream& os, version_t v)
{
	return os << v.major << '.' << v.minor;
}

inline bool operator==(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() == rhs.operator std::pair<int, int>();
}

inline bool operator!=(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() != rhs.operator std::pair<int, int>();
}

inline bool operator<(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() < rhs.operator std::pair<int, int>();
}

inline bool operator<=(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() <= rhs.operator std::pair<int, int>();
}

inline bool operator>(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() > rhs.operator std::pair<int, int>();
}

inline bool operator>=(const version_t& lhs, const version_t& rhs)
{
	return lhs.operator std::pair<int, int>() >= rhs.operator std::pair<int, int>();
}


namespace version_numbers {

/**
 * This "value" is what the Runtime API returns if no version is supported by the driver
 *
 * @note this is super-ugly, I'd rather n ot use  it at all
 */
constexpr version_t none()
{
	return { 0, 0 };
}

inline version_t make(combined_version_t combined_version)
{
	return version_t::from_single_number(combined_version);
}
inline version_t make(int major, int minor)
{
	return { major, minor };
}

/**
 * Obtains the maximum version of the CUDA Runtime supported by the
 * driver currently loaded by the operating system
 *
 * @todo In future CUDA versions which support C++17 - return
 * an optional
 *
 * @return If an nVIDIA GPU driver is installed on this system,
 * the maximum CUDA version it supports is returned.
 * If no version is supported, @ref version_numbers::none() is returned.
 */
inline version_t maximum_supported_by_driver() {
	combined_version_t version;
	auto status = cudaDriverGetVersion(&version);
	throw_if_error(status, "Failed obtaining the maximum CUDA version supported by the nVIDIA GPU driver");
	return version_t::from_single_number(version);
}

/**
 * Obtains the CUDA Runtime version
 *
 * @note unlike {@ref maximum_supported_by_driver()}, 0 cannot be returned,
 * as we are actually using the runtime to obtain the version, so it does
 * have _some_ version.
 */
inline version_t runtime() {
	combined_version_t version;
	auto status = cudaRuntimeGetVersion(&version);
	throw_if_error(status, "Failed obtaining the CUDA runtime version");
	return version_t::from_single_number(version);
}

} // namespace version_numbers
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_VERSIONS_HPP_ */
