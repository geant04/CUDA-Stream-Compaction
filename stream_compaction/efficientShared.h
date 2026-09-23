#pragma once
#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace EfficientShared {
        StreamCompaction::Common::PerformanceTimer& timer();

        float getGpuTime();
        void scan_efficient_shared_naive(int n, int *odata, const int *idata);
        void scan_efficient_warp_shared(int n, int *odata, const int *idata);
    }
}
