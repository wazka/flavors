/**
 * @file api_wrappers.h
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers && related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_H_
#define CUDA_API_WRAPPERS_H_

#include <types.h>
#include <constants.h>
#include <error.hpp>
#include <versions.hpp>
#include <kernel_launch.cuh>
#include <device_properties.hpp>
#include <pci_id.hpp>
#include <device_count.hpp>
#include <current_device.hpp>
#include <device_function.hpp>
#include <memory.hpp>
#include <pointer.hpp>
#include <unique_ptr.hpp>
#include <ipc.hpp>

#include <stream.hpp>
#include <device.hpp>
#include <event.hpp>

#include <multi_wrapper_impls.hpp>

#endif /* CUDA_API_WRAPPERS_H_ */
