#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#include <iostream>
#include <vector>

namespace StreamCompaction {
    namespace EfficientShared {
        using StreamCompaction::Common::PerformanceTimer;

#define MAX(x, y) ((x < y) ? y : x)

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARP_PASSES 5
#define WARP_SCAN_SIZE MAX(BLOCK_SIZE / WARP_SIZE, 1)
#define FULL_MASK 0xFFFFFFFF

        enum ScanType : int {
            NaiveShared = 0,
            WarpShared = 1
        };

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        float getGpuTime()
        {
            return timer().getGpuElapsedTimeForPreviousOperation();
        }

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
        
        __device__ static void blockInclusiveToExclusiveInternal(int n, int *writeBuffer, int *readBuffer)
        {
            int index = threadIdx.x;
            if (index >= n)
            {
                return;
            }

            if (index == 0)
            {
                writeBuffer[index] = 0;
                return;
            }

            writeBuffer[index] = readBuffer[index - 1];
        }

        __device__ void deviceBlockInclusiveToExclusive(int n, int *writeBuffer, int *readBuffer)
        {
            blockInclusiveToExclusiveInternal(n, writeBuffer, readBuffer);
        }

        __global__ void blockInclusiveToExclusive(int n, int *writeBuffer, int *readBuffer)
        {
            blockInclusiveToExclusiveInternal(n, writeBuffer, readBuffer);
        }

        __device__ int warpInternalScan(int laneValue)
        {
            unsigned int mask = FULL_MASK;
            unsigned int logOffset;
            unsigned int laneId = threadIdx.x % WARP_SIZE;

            for ( logOffset = 0; logOffset <= 4; logOffset++ )
            {
                unsigned int delta = 1 << logOffset;
                unsigned int readValue = __shfl_up_sync(mask, laneValue, delta, WARP_SIZE);

                if (laneId >= delta)
                {
                    laneValue += readValue;
                }
            }

            return laneValue;
        }

        __device__ void blockUsingWarpSumInternalScan(const int n, const int passes, const bool isInclusive, int *dev_odata, int *dev_idata)
        {
            // Warp sum arrays
            __shared__ int warpSums[WARP_SCAN_SIZE];

            int localThreadId = threadIdx.x;
            int laneId = localThreadId % WARP_SIZE;
            int warpId = localThreadId / WARP_SIZE;

            int globalThreadId = localThreadId + blockIdx.x * BLOCK_SIZE;
            int threadValue = (globalThreadId < n) ? dev_idata[globalThreadId] : 0;
            int warpScanOutput = warpInternalScan(threadValue);

            // Early return, this would occur if n <= 32
            // We would've performed the scan using intrinsics only
            if (passes <= WARP_PASSES)
            {
                if (globalThreadId < n)
                {
                    int difference = isInclusive ? 0 : threadValue;
                    dev_odata[globalThreadId] = warpScanOutput - difference;
                }
                return;
            }

            if (laneId == WARP_SIZE - 1)
            {
                warpSums[warpId] = warpScanOutput;
            }

            // Sync after warp scan results are populated
            __syncthreads();

            // Perform internal, exclusive scan on list of warp sums
            if (warpId == 0)
            {
                int warpSum = (laneId < WARP_SCAN_SIZE) ? warpSums[laneId] : 0;
                int warpSumScanOutput = warpInternalScan(warpSum);

                if (laneId < WARP_SCAN_SIZE)
                {
                    warpSums[laneId] = warpSumScanOutput;
                }
            }

            __syncthreads();

            // Add "exclusive-scan" results to the respective warps
            unsigned int difference = (warpId == 0) ? 0 : warpSums[warpId - 1];
            difference += isInclusive ? 0 : -threadValue;

            if (globalThreadId < n)
            {
                dev_odata[globalThreadId] = warpScanOutput + difference;
            }
        }

        __device__ void internalScanPass(int n, int stride, int *writeBuffer, int *readBuffer)
        {
            int localThreadId = threadIdx.x;
            
            if (localThreadId < n)
            {
                int out = readBuffer[localThreadId];
                if (localThreadId >= stride)
                {
                    out += readBuffer[localThreadId - stride];
                }
                writeBuffer[localThreadId] = out;
            }
        }

        __device__ void blockInternalScan(int n, int passes, int *dev_odata, int *dev_idata)
        {
            __shared__ int read[BLOCK_SIZE];
            __shared__ int write[BLOCK_SIZE];

            int localThreadId = threadIdx.x;
            read[localThreadId] = (localThreadId < n) ? dev_idata[localThreadId + blockIdx.x * blockDim.x] : 0;

            // Sync threads 1
            __syncthreads();

            int *readBuffer = read;
            int *writeBuffer = write;
            for (int pass = 0; pass < passes; pass++)
            {
                int stride = 1 << pass;
                internalScanPass(n, stride, writeBuffer, readBuffer);

                // Sync threads log(n) times
                __syncthreads();

                int *temp = readBuffer;
                readBuffer = writeBuffer;
                writeBuffer = temp;
            }

            // Add incluse/exclusive result stuff
            deviceBlockInclusiveToExclusive(n, writeBuffer, readBuffer);

            // Final write out
            dev_odata[localThreadId] = writeBuffer[localThreadId];
        }

        __global__ void efficientSharedScan(int scanType, int n, int passes, const bool isInclusive, int *dev_odata, int *dev_idata)
        {
            if (scanType == 0)
            {
                // This method only works if the number of elements can be processed by one block.
                // Exists as a stepping stone to the warp-based scan method that works for multi-block
                blockInternalScan(n, passes, dev_odata, dev_idata);
                return;
            }

            if (scanType == 1)
            {
                blockUsingWarpSumInternalScan(n, passes, isInclusive, dev_odata, dev_idata);
                return;
            }
        }

        __global__ void writeBlockSumsToArray(int n, int blocks, const bool isInclusive, int *dev_scanned_idata, int *dev_block_sum_array, int *dev_idata)
        {
            // Each thread maps to 1 block directly
            int globalThreadId = threadIdx.x + blockIdx.x * BLOCK_SIZE;
            int blockId = globalThreadId;

            if (blockId >= blocks)
            {
                return;
            }
            else
            {
                int globalBlockLastElementId = (blockId + 1) * BLOCK_SIZE - 1;
                
                // if threadId exceeds n, last element should be n-1
                globalBlockLastElementId = (globalBlockLastElementId > n ? n - 1 : globalBlockLastElementId);

                int difference = isInclusive ? 0 : dev_idata[globalBlockLastElementId];
                dev_block_sum_array[blockId] = dev_scanned_idata[globalBlockLastElementId] + difference;
            }
        }

        __global__ void addScannedSumsToBlocks(int n, int *dev_scanned_block_sum_array, int *dev_odata)
        {
            int blockId = blockIdx.x;
            int localThreadId = threadIdx.x;
            int globalThreadId = localThreadId + blockId * BLOCK_SIZE;

            if (globalThreadId >= n)
            {
                return;
            }
            else
            {
                int scannedBlockSum = dev_scanned_block_sum_array[blockId];
                dev_odata[globalThreadId] += scannedBlockSum;
            }
        }

        static void multi_pass_block_scan(int n, int *dev_odata, int *dev_idata, const bool isInclusive)
        {
            const int passes = ilog2ceil(n);
            const int paddedArraySize = 1 << passes;
            const int blocks = (paddedArraySize + BLOCK_SIZE - 1) / BLOCK_SIZE;

            if (blocks <= 1)
            {
                // We'll just make everything inclusive by default. I guess. this is sort of rough.
                efficientSharedScan<<<blocks, BLOCK_SIZE>>>(ScanType::WarpShared, n, passes, isInclusive, dev_odata, dev_idata);
                return;
            }
            else
            {
                // Scratch buffer allocation for temp block sums
                const int blockSumsSizeInBytes = sizeof(int) * blocks;
                const int blockSumArraySize = (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
                
                int *dev_block_sums;
                int *dev_block_scanned_sums;
                cudaMalloc((void**)&dev_block_sums, blockSumsSizeInBytes);
                cudaMalloc((void**)&dev_block_scanned_sums, blockSumsSizeInBytes);
                cudaMemset(dev_block_sums, 0, blockSumsSizeInBytes);
                cudaMemset(dev_block_scanned_sums, 0, blockSumsSizeInBytes);

                // Perform scan on the block-level, in addition to gathering sums from each block
                efficientSharedScan<<<blocks, BLOCK_SIZE>>>(ScanType::WarpShared, n, passes, isInclusive, dev_odata, dev_idata);

                // Gather block sums
                writeBlockSumsToArray<<<blockSumArraySize, BLOCK_SIZE>>>(n, blocks, isInclusive, dev_odata, dev_block_sums, dev_idata );

                // Recursively perform exclusive scan on the sum, use false param
                multi_pass_block_scan(blocks, dev_block_scanned_sums, dev_block_sums, false);

                // Propogate results back to the blocks from the dev_block_scanned_sums... results should be made exclusive but
                // to replicate slide results, we'll just make them inclusive
                addScannedSumsToBlocks<<<blocks, BLOCK_SIZE>>>(n, dev_block_scanned_sums, dev_odata);

                cudaFree(dev_block_sums);
                cudaFree(dev_block_scanned_sums);
            }
        }

        void scan_internal(const int scanType, int n, int *odata, const int *idata) {
            int* dev_idata;
            int* dev_odata;
            const int arraySize = n;
            const int passes = ilog2ceil(arraySize);
            const int paddedArraySize = 1 << passes;
            const int sizeInBytes = sizeof(int) * paddedArraySize;

            // Memory allocation
            {
                cudaMalloc((void**)&dev_idata, sizeInBytes);
                cudaMalloc((void**)&dev_odata, sizeInBytes);
                cudaMemcpy(dev_idata, idata, sizeInBytes, cudaMemcpyHostToDevice);
                timer().startGpuTimer();
            }

            // Stream compaction algorithm/dispatches
            // 1D grid of blocks for stream compaction
            {   
                 multi_pass_block_scan(arraySize, dev_odata, dev_idata, false);
            }

            // Send results back to host + cleanup
            {
                timer().endGpuTimer();
                cudaMemcpy(odata, dev_odata, sizeInBytes, cudaMemcpyDeviceToHost);
                cudaFree(dev_idata);
                cudaFree(dev_odata);
            }
        }

        void scan_efficient_shared_naive(int n, int *odata, const int *idata)
        {
            scan_internal(0, n, odata, idata);
        }

        void scan_efficient_warp_shared(int n, int *odata, const int *idata)
        {
            scan_internal(1, n, odata, idata);
        }
    }
}
