/**
 * @file device_properties.hpp
 *
 * @brief Classes for holding CUDA device properties and
 * CUDA compute capability values.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_
#define CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_

#include "types.h"
#include "constants.h"
#include "pci_id.h"

#include <cuda_runtime_api.h>

#include <string>
#include <unordered_map>
#include <utility>

namespace cuda {

namespace device {

/**
 * A numeric designator of an architectural generation of CUDA devices
 *
 * @note See @url https://en.wikipedia.org/wiki/Volta_(microarchitecture)
 * && previous architectures' pages via "previous" links.
 * Also see @ref compute_capability_t .
 */
struct compute_architecture_t {
	/**
	 * A @ref compute_capability_t has a "major" && a "minor" number,
	 * with "major" indicating the architecture; so this struct only
	 * has a "major" numner
	 */
	unsigned major;

	static const char* name(unsigned major_compute_capability_version)
	{
		static std::unordered_map<unsigned, std::string> arch_names =
		{
			{ 1, "Tesla"   },
			{ 2, "Fermi"   },
			{ 3, "Kepler"  },
			{ 5, "Maxwell" },
			{ 6, "Pascal"  },
			{ 7, "Pascal"  },
		};
		return arch_names.at(major_compute_capability_version).c_str();
			// Will throw for invalid architecture numbers!
	}
	unsigned max_warp_schedulings_per_processor_cycle() const
	 {
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 1, 1 },
			{ 2, 2 },
			{ 3, 4 },
			{ 5, 4 },
			{ 6, 2 },
			{ 7, 2 }, // speculation
		};
		return data.at(major);
	}
	
	unsigned max_resident_warps_per_processor() const 
	{
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 1, 24 },
			{ 2, 48 },
			{ 3, 64 },
			{ 5, 64 },
			{ 6, 64 },
			{ 7, 64 },
		};
		return data.at(major);
	}

	unsigned max_in_flight_threads_per_processor() const
	{
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 1,   8 },
			{ 2,  32 },
			{ 3, 192 },
			{ 5, 128 },
			{ 6, 128 },
			{ 7, 128 }, // speculation
		};
		return data.at(major);
	}
	/**
	 * @note On some architectures, the shared memory / L1 balance is configurable,
	 * so you might not Get the maxima here without making this configuration
	 * setting
	 */
	shared_memory_size_t max_shared_memory_per_block() const
	{
		enum : shared_memory_size_t { KiB = 1024 };
		// On some architectures, the shared memory / L1 balance is configurable,
		// so you might not Get the maxima here without making this configuration
		// setting
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 1,  16 * KiB },
			{ 2,  48 * KiB },
			{ 3,  48 * KiB },
			{ 5,  64 * KiB },
			{ 6,  64 * KiB },
			{ 7,  96 * KiB },
				// this is a speculative figure based on:
				// https://devblogs.nvidia.com/parallelforall/inside-volta/
		};
		return data.at(major);
	}

	const char* name() const { return name(major); }

	bool is_valid() const
	{
		return (major > 0) && (major < 9999); // Picked this up from the CUDA code somwhere
	}

};

inline bool operator ==(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major == rhs.major;
}
inline bool operator !=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major != rhs.major;
}
inline bool operator <(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major < rhs.major;
}
inline bool operator <=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major < rhs.major;
}
inline bool operator >(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major > rhs.major;
}
inline bool operator >=(const compute_architecture_t& lhs, const compute_architecture_t& rhs)
{
	return lhs.major > rhs.major;
}


// TODO: Consider making this a non-POD struct,
// with a proper ctor checking validity, an operator converting to pair etc;
// however, that would require including at least std::utility, if not other
// stuff (e.g. for an std::hash specialization)
/**
 * A numeric designator of the computational capabilities of a CUDA device
 *
 * @note See @url https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications
 * for a specification of capabilities by CC values
 */
struct compute_capability_t {
	unsigned major;
	unsigned minor;

	unsigned as_combined_number() const { return major * 10 + minor; }
	shared_memory_size_t max_shared_memory_per_block() const
	{
		enum : shared_memory_size_t { KiB = 1024 };
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 37, 112 * KiB },
			{ 52,  96 * KiB },
			{ 61,  96 * KiB },
		};
		auto cc = as_combined_number();
		auto it = data.find(cc);
		if (it != data.end()) { return it->second; }
		return architecture().max_shared_memory_per_block();
	}

	unsigned max_resident_warps_per_processor() const {
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 11, 24 },
			{ 12, 32 },
			{ 13, 32 },
		};
		auto cc = as_combined_number();
		auto it = data.find(cc);
		if (it != data.end()) { return it->second; }
		return architecture().max_resident_warps_per_processor();
	}

	unsigned max_warp_schedulings_per_processor_cycle() const {
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 61, 4 },
			{ 62, 4 },
		};
		auto cc = as_combined_number();
		auto it = data.find(cc);
		if (it != data.end()) { return it->second; }
		return architecture().max_warp_schedulings_per_processor_cycle();
	}

	unsigned max_in_flight_threads_per_processor() const {
		static std::unordered_map<unsigned, unsigned> data =
		{
			{ 21,  48 },
			{ 60,  64 },
		};
		auto cc = as_combined_number();
		auto it = data.find(cc);
		if (it != data.end()) { return it->second; }
		return architecture().max_in_flight_threads_per_processor();
	}

	compute_architecture_t architecture() const { return compute_architecture_t { major }; }

	bool is_valid() const
	{
		return (major > 0) && (major < 9999) && (minor > 0) && (minor < 9999);
			// Picked this up from the CUDA code somwhere
	}

	static compute_capability_t from_combined_number(unsigned combined)
	{
		return  { combined / 10, combined % 10 };
	}
};

inline bool operator ==(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major == rhs.major && lhs.minor == rhs.minor;
}
inline bool operator !=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major != rhs.major || lhs.minor != rhs.minor;
}
inline bool operator <(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major < rhs.major || (lhs.major == rhs.major && lhs.minor < rhs.minor);
}
inline bool operator <=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major < rhs.major || (lhs.major == rhs.major && lhs.minor <= rhs.minor);
}
inline bool operator >(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major > rhs.major || (lhs.major == rhs.major && lhs.minor > rhs.minor);
}
inline bool operator >=(const compute_capability_t& lhs, const compute_capability_t& rhs)
{
	return lhs.major > rhs.major || (lhs.major == rhs.major && lhs.minor >= rhs.minor);
}

inline compute_capability_t make_compute_capability(unsigned combined)
{
	return compute_capability_t::from_combined_number(combined);
}

inline compute_capability_t make_compute_capability(unsigned major, unsigned minor)
{
	return { major, minor };
}

/**
 * @brief A structure holding a collection various properties of a device
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties && flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays && limited-length strings (see
 * @ref cuda::device::properties_t), && flags are either binary or
 * small-finite-domain type fitting into an overall flagss value (see
 * @ref cuda::device_t::flags_t). Flags && properties are obtained all at once,
 * attributes are more one-at-a-time.
 *
 */
struct properties_t : public cudaDeviceProp {

	properties_t() = default;
	properties_t(cudaDeviceProp& cdp) : cudaDeviceProp(cdp) { };
	bool usable_for_compute() const
	{
		return computeMode != cudaComputeModeProhibited;
	}
	compute_capability_t compute_capability() const { return { (unsigned) major, (unsigned) minor }; }
	compute_architecture_t compute_architecture() const { return { (unsigned) major }; };
	pci_location_t pci_id() const { return { pciDomainID, pciBusID, pciDeviceID }; }

	unsigned long long max_in_flight_threads_on_device() const
	{
		return compute_capability().max_in_flight_threads_per_processor() * multiProcessorCount;
	}

	grid_block_dimension_t max_threads_per_block() const { return maxThreadsPerBlock; }
	grid_block_dimension_t max_warps_per_block() const { return maxThreadsPerBlock / warp_size; }
	bool can_map_host_memory() const { return canMapHostMemory != 0; }
};

} // namespace device
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_PROPERTIES_HPP_ */
