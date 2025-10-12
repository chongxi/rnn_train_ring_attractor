#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <iostream>
#include <vector>

namespace cg = cooperative_groups;