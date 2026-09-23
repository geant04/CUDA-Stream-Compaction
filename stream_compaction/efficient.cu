#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#include <iostream>

// Debugging defines, toggling off/on chunks of code helps me figure stuff out
#define SIMPLE_EFFICIENT_SCAN 1
#define ENABLE_DOWNSWEEP 1

namespace StreamCompaction {
    namespace Efficient {
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

        __global__ void upsweep(int n, int stride, int* dev_odata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n || index >= (n / stride / 2))
            {
                return;
            }

            // stride = 2^d, d \in [0, log2(n) - 1]
            int strideMult2 = stride * 2;
            if (index < (n / strideMult2))
            {
                if ((n/strideMult2) == 1)
                {
                    // Last element clear at final stage
                    dev_odata[index + strideMult2 - 1] = 0;
                    return;
                }

                int writeIndex = index * strideMult2;
                dev_odata[writeIndex + strideMult2 - 1] += dev_odata[writeIndex + stride - 1];
            }
        }

        __global__ void downsweep(int n, int stride, int* dev_odata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n || index >= (n / stride))
            {
                return;
            }

            int strideDiv2 = stride / 2;

            if (index < (n / stride))
            {
                int writeIndex = index * stride;

                int left = dev_odata[writeIndex + strideDiv2 - 1];
                int right = dev_odata[writeIndex + stride - 1];

                dev_odata[writeIndex + strideDiv2 - 1] = right;
                dev_odata[writeIndex + stride - 1] += left;
            }
        }

        // Shared mem implementation
        __global__ void optimizedSharedScan(int n, int* dev_odata)
        {
            extern __shared__ float sharedData[];
            
            int threadID = threadIdx.x;

            // This only works for thread up to max block size for now.
            // I'll have to spend a few hours to figure out how to fit everything on a SM.
            // There are also some potential bank conflicts.
            sharedData[threadID] = dev_odata[threadID];

            __syncthreads();

            // Shared mem upsweep, nothing too different.
            for (int d = n/2; d > 0; d >>= 1)
            {
                if (threadID < d)
                {
                    int strideMult2 = n/d;
                    int stride = strideMult2 >> 1;

                    int writeIndex = threadID * strideMult2;
                    sharedData[writeIndex + strideMult2 - 1] += sharedData[writeIndex + stride - 1];
                }

                __syncthreads();
            }

            // clear last element... this will waste a few cycles
            if (threadID < 1)
            {
                sharedData[n - 1] = 0;
            }

            __syncthreads();

#if ENABLE_DOWNSWEEP
            for (int d = 1; d < n; d <<= 1)
            {
                if (threadID < d)
                {
                    int stride = n/d;
                    int strideDiv2 = stride >> 1;

                    int writeIndex = threadID * stride;

                    int left = sharedData[writeIndex + strideDiv2 - 1];
                    int right = sharedData[writeIndex + stride - 1];

                    sharedData[writeIndex + strideDiv2 - 1] = right;
                    sharedData[writeIndex + stride - 1] += left;
                }

                __syncthreads();
            }

            __syncthreads();
#endif


            dev_odata[threadID] = sharedData[threadID];
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        void scanDispatch(int blocks, int blockSize, int n, int stages, int &stride, int *dev_odata)
        {
            // upsweep, write to temp buffer
            for (int d = 0; d <= stages; d++)
            {
                upsweep<<<blocks, blockSize>>>(n, stride, dev_odata);
                stride <<= 1;
            }

            for (int d = 0; d <= stages; d++)
            {
                downsweep<<<blocks, blockSize>>>(n, stride, dev_odata);
                stride >>= 1;
            }
        }


        void scan(int n, int *odata, const int *idata) {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_odata;

            int sizeOfData = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_odata, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = PROFILING_BLOCK_SIZE;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2(paddedN) - 1;
            int stride = 1;

            timer().startGpuTimer();
            
            scanDispatch(blocks, blockSize, paddedN, stages, stride, dev_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            
            cudaFree(dev_odata);
        }


        void optimizedScanDispatch(int paddedN, int* dev_odata)
        {
            int blockSize = PROFILING_BLOCK_SIZE;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2(paddedN);
            int stride = 1;

            int optimizedBlockSize = blockSize;
            int optimizedBlocks = blocks;

            int threadsToRun, optimizedN;

            for (int d = 0; d < stages; d++)
            {
                upsweep<<<optimizedBlocks, optimizedBlockSize>>>(paddedN, stride, dev_odata);
                stride <<= 1;

                // Threads to run are halved
                threadsToRun = paddedN >> (d + 1);
                optimizedN = threadsToRun;

                threadsToRun = (threadsToRun >= blockSize) ? blockSize : threadsToRun;
                threadsToRun = (threadsToRun <= 32) ? 32 : threadsToRun;

                optimizedBlockSize = threadsToRun;
                optimizedBlocks = (optimizedN + optimizedBlockSize - 1) / optimizedBlockSize;
            }

#if ENABLE_DOWNSWEEP
            for (int d = 0; d < stages; d++)
            {
                downsweep<<<optimizedBlocks, optimizedBlockSize>>>(paddedN, stride, dev_odata);
                stride >>= 1;

                // Since N is padded to the nearest power of 2, this logic to compute # of threads is fine
                threadsToRun = 1u << (d + 1);
                optimizedN = threadsToRun;

                threadsToRun = (threadsToRun <= 32) ? 32 : threadsToRun;
                threadsToRun = (threadsToRun >= blockSize) ? blockSize : threadsToRun;

                optimizedBlockSize = threadsToRun;
                optimizedBlocks = (optimizedN + optimizedBlockSize - 1) / optimizedBlockSize;
            }
#endif
        }

        void optimizedScan(int n, int *odata, const int *idata)
        {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_odata;
            int sizeOfData = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_odata, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            optimizedScanDispatch(paddedN, dev_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            
            cudaFree(dev_odata);
        }

        // ------------------------------------------------------------------------------------------
        __global__ void optimizedUpDownsweep(int blockSize, int *dev_odata, int *dev_blockSums)
        {
            extern __shared__ float sharedData[];
            
            // local ID in the thread
            int threadID = threadIdx.x;

            sharedData[threadID] = dev_odata[blockDim.x * blockIdx.x + threadID];

            __syncthreads();

            // Shared mem upsweep
            for (int d = blockSize >> 1; d > 0; d >>= 1)
            {
                if (threadID < d)
                {
                    int strideMult2 = blockSize / d;
                    int stride = strideMult2 >> 1;

                    int writeIndex = threadID * strideMult2;
                    sharedData[writeIndex + strideMult2 - 1] += sharedData[writeIndex + stride - 1];
                }

                __syncthreads();
            }

            __syncthreads();

            // Add the blockSum, and then clear the last element
            int blockSum = sharedData[blockSize - 1];

            if (threadID < 1)
            {
                dev_blockSums[blockIdx.x] = blockSum;
                sharedData[blockSize - 1] = 0;
            }

            __syncthreads();

            // Downsweep, result gives us an EXCLUSIVE SCAN.
            for (int d = 1; d < blockSize; d <<= 1)
            {
                if (threadID < d)
                {
                    int stride = blockSize/d;
                    int strideDiv2 = stride >> 1;

                    int writeIndex = threadID * stride;

                    int left = sharedData[writeIndex + strideDiv2 - 1];
                    int right = sharedData[writeIndex + stride - 1];

                    sharedData[writeIndex + strideDiv2 - 1] = right;
                    sharedData[writeIndex + stride - 1] += left;
                }

                __syncthreads();
            }

            __syncthreads();

            // By this point, we have an exclusive scan, but we need an inclusive scan for our overall scan to work.
            // Then, we'll add back the blockSum and then life is good.

            // For the given ID, we need to shift left to prepare for the inclusive scan.
            // Last element will be 0 consequently, I think.

            if (threadID == blockSize - 1)
            {
                dev_odata[blockDim.x * blockIdx.x + threadID] = blockSum;
            }
            else
            {
                dev_odata[blockDim.x * blockIdx.x + threadID] = sharedData[threadID + 1];
            }
        }

        __global__ void addBlockSums(int *dev_blockSums, int *dev_odata)
        {            
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            int blockSum = dev_blockSums[blockIdx.x];

            dev_odata[index] += blockSum;
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

        void optimizedMemScan(int n, int *odata, const int *idata)
        {
            int paddedN = 1 << ilog2ceil(n);

            int *dev_odata2;
            int *dev_odata;
            int *dev_blockSums;

            int sizeOfData = paddedN * sizeof(int);
            cudaMalloc((void**)&dev_odata, sizeOfData);
            cudaMalloc((void**)&dev_odata2, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int finalBlockSize = 128;
            int finalBlocks = (paddedN + finalBlockSize - 1) / finalBlockSize;

            int paddedBlocks = 1 << ilog2ceil(blocks);
            int blockSumByteSize = paddedBlocks * sizeof(int);

            cudaMalloc((void**)&dev_blockSums, blockSumByteSize);
            cudaMemset(dev_blockSums, 0, blockSumByteSize);

            // insert gpu timer start
            timer().startGpuTimer();

            // step 1: upsweep sum accumulate on the input data
            optimizedUpDownsweep<<<blocks, blockSize>>>(blockSize, dev_odata, dev_blockSums);

            // step 2: perform exclusive scan on BLOCK sum
            optimizedScanDispatch(paddedBlocks, dev_blockSums);

            // step 3: add the blockSums back into the BLOCKS
            addBlockSums<<<blocks, blockSize>>>(dev_blockSums, dev_odata);

            // step 4: final dispatch to convert from inclusive to exclusive
            inclusiveToExclusive<<<finalBlocks, finalBlockSize>>>(paddedN, dev_odata2, dev_odata);
            
            // insert gpu timer end
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata2, sizeOfData, cudaMemcpyDeviceToHost);
            
            cudaFree(dev_odata2);
            cudaFree(dev_odata);
            cudaFree(dev_blockSums);
        }

        void optimizedMemScanOld(int n, int *odata, const int *idata)
        {
            // LOL!!!
            if (n >= (1 << 8))
            {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);

            int *dev_odata;

            int sizeOfData = paddedN * sizeof(int);

            cudaMalloc((void**)&dev_odata, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Ideally, we use one kernel per thread to reduce overhead from running MULTIPLE kernel dispatches.
            // This also takes care of early terminating warps early on, as we don't run more than 1 dispatch.
            optimizedSharedScan<<<blocks, blockSize>>>(paddedN, dev_odata);
            
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            
            cudaFree(dev_odata);
        }


        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */

        __global__ void kernMapToBoolean(int n, int *dev_bitmap)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
            {
                return;
            }

            dev_bitmap[index] = (dev_bitmap[index] > 0) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int *dev_bitmap, int *dev_odata, int *dev_idata)
        {
            int index = blockDim.x * blockIdx.x + threadIdx.x;

            if (index >= n)
            {
                return;
            }

            if (dev_bitmap[index] > 0)
            {
                int scatterIndex = dev_odata[index];
                dev_odata[scatterIndex] = dev_idata[index];
            }
        }

        int compact(int n, int *odata, const int *idata) 
        {
            int paddedN = 1 << ilog2ceil(n);
            int sizeOfData = paddedN * sizeof(int);

            int *dev_idata;
            int *dev_odata;
            int *dev_bitmap;

            cudaMalloc((void**)&dev_idata, sizeOfData);
            cudaMalloc((void**)&dev_odata, sizeOfData);
            cudaMalloc((void**)&dev_bitmap, sizeOfData);

            // Copy idata to dev_odata first, this way we can easily modify in place
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_bitmap, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (paddedN + blockSize - 1) / blockSize;

            int stages = ilog2(paddedN) - 1;
            int stride = 1;

            timer().startGpuTimer();

            // Write to dev_bitmap, input is idata memcpyed to dev_bitmap
            kernMapToBoolean<<<blocks, blockSize>>>(paddedN, dev_bitmap);

            // Copy dev_bitmap info to dev_odata, this is needed so we can run scan
            cudaMemcpy(dev_odata, dev_bitmap, sizeOfData, cudaMemcpyDeviceToDevice);

            // Scan writes output to dev_odata
            scanDispatch(blocks, blockSize, paddedN, stages, stride, dev_odata);

            // Write to dev_odata with inputs idata, bitmap, and scanOutput, which is dev_odata at this point.
            kernScatter<<<blocks, blockSize>>>(paddedN, dev_bitmap, dev_odata, dev_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, sizeOfData, cudaMemcpyDeviceToHost);
            cudaFree(dev_odata);
            cudaFree(dev_idata);
            cudaFree(dev_bitmap);

            return odata[paddedN - 1];
        }
    }
}
