/**
 * @file api_wrappers.h
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * Runtime API wrappers && related headers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_H_
#define CUDA_API_WRAPPERS_H_

#include "api/types.h"
#include "api/constants.h"
#include "api/error.hpp"
#include "api/versions.hpp"
#include "api/kernel_launch.cuh"
#include "api/device_properties.hpp"
#include "api/pci_id.hpp"
#include "api/device_count.hpp"
#include "api/current_device.hpp"
#include "api/device_function.hpp"
#include "api/memory.hpp"
#include "api/pointer.hpp"
#include "api/unique_ptr.hpp"
#include "api/ipc.hpp"
#include "api/stream.hpp"
#include "api/device.hpp"
#include "api/event.hpp"
#include "api/multi_wrapper_impls.hpp"

#endif /* CUDA_API_WRAPPERS_H_ */
