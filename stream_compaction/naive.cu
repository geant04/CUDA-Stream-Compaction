#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        float getGpuTime()
        {
            return timer().getGpuElapsedTimeForPreviousOperation();
        }

        // TODO: __global__
        __global__ void naiveScan(int n, int stride, int *dev_odata, int *dev_idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index >= stride)
            {
                int out = dev_idata[index - stride] + dev_idata[index];

                dev_odata[index] = out;
            }
            else
            {
                dev_odata[index] = dev_idata[index];
            }
        }
 
        __global__ void inclusiveToExclusive(int n, int *dev_odata, int *dev_idata)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index == 0)
            {
                dev_odata[index] = 0;
                return;
            }

            dev_odata[index] = dev_idata[index - 1];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int paddedN = 1 << ilog2ceil(n);

            // Cuda device malloc set up using host data, should I move all of this before the startGpuTimer?
            int *dev_odata;
            int *dev_idata;
            size_t dataSize = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_odata, dataSize);
            cudaMalloc((void**)&dev_idata, dataSize);

            cudaMemcpy(dev_idata, idata, dataSize, cudaMemcpyHostToDevice);

            // Kernel dispatches
            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2ceil(paddedN);
            int stride = 1;

            timer().startGpuTimer();
            for (int i = 1; i <= stages + 1; i++)
            {
                naiveScan<<<blocks, blockSize>>>(paddedN, stride, dev_odata, dev_idata);
                
                // ping pong
                std::swap(dev_odata, dev_idata);

                stride <<= 1;
            }

            inclusiveToExclusive<<<blocks, blockSize>>>(paddedN, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, dataSize, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
        }
    }
}
