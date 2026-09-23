#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction//efficientShared.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <functional>

const int SIZE = 1 << 24;
const int NPOT = SIZE - 3;
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

#define DISABLE_CPU 0
#define ENABLE_NON_POWER_OF_TWO 1
#define USE_OPTIMIZED 1

#define PROFILING 0
#define PROFILE_NON_POWER_OF_TWO 1

struct TestParams
{
    const int arraySize;
    int* inArray;
    int* outArray;
};

void testScanPow2(
    TestParams* testParams,
    const char* msg, 
    void (*scanFunction)(int, int*, const int*),
    float (*getTimeFunction)())
{
    zeroArray(testParams->arraySize, testParams->outArray);
    printDesc(msg);
    scanFunction( testParams->arraySize, testParams->outArray, testParams->inArray );
    printElapsedTime( getTimeFunction(), "(CUDA Measured)");

    int* groundTruth = b;
    if (printCmpResult(SIZE, groundTruth, testParams->outArray) )
    {
        printArray(SIZE, testParams->outArray, true);
    }
}

int main(int argc, char* argv[]) {
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    int* groundTruth = b;
    int* generatedArrayData = a;
    int* outArrayData = c;
    TestParams testParams = { SIZE, generatedArrayData, outArrayData };

    {
        // Test case generation
        // Leave a 0 at the end to test that edge case
        genArray(SIZE - 1, a, 50);
        generatedArrayData[SIZE - 1] = 0;
        printArray(SIZE, generatedArrayData, true);

        {
            zeroArray(SIZE, groundTruth);
            printDesc("cpu scan, power-of-two");
            StreamCompaction::CPU::scan(SIZE, groundTruth, generatedArrayData);
            printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
            printArray(SIZE, groundTruth, true);
        }

        // TODO: do some neat sorting tests
        // TODO: do some neat mat-mul kernel tests
    }

    // Test scan with GPU implementation
    {
        testScanPow2( &testParams, "naive scan, power-of-two", 
            &StreamCompaction::Naive::scan, &StreamCompaction::Naive::getGpuTime );
        testScanPow2( &testParams, "work-efficient scan, power-of-two", 
            &StreamCompaction::Efficient::scan, &StreamCompaction::Efficient::getGpuTime );
        testScanPow2( &testParams, "thrust scan, power-of-two", 
            &StreamCompaction::Thrust::scan, &StreamCompaction::Thrust::getGpuTime );
        testScanPow2( &testParams, "efficient warp shared scan, power-of-two", 
            &StreamCompaction::EfficientShared::scan_efficient_warp_shared, &StreamCompaction::EfficientShared::getGpuTime );
    }

    
    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}



// -------------------------------- LEGACY CODE DOWN BELOW -------------------------------

void getAvgTest(int trials)
{
    if (trials < 0)
    {
        return;
    }

    float avgCPUTime = 0.0f;
    float avgNaiveTime = 0.0f;
    float avgWorkEfficientTime = 0.0f;
    float avgOptWorkEfficientTime = 0.0f;
    float avgOptSharedWorkEfficientTime = 0.0f;
    float avgThrustTime = 0.0f;

    int testSize = SIZE;

#if PROFILE_NON_POWER_OF_TWO
    testSize = NPOT;
#endif

    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, b);
        StreamCompaction::CPU::scan(testSize, b, a);
        avgCPUTime += StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation();
    }
    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, c);
        StreamCompaction::Naive::scan(testSize, c, a);
        avgNaiveTime += StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation();
    }
        
    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, c);
        StreamCompaction::Efficient::scan(testSize, c, a);
        avgWorkEfficientTime += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }

    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, c);
        StreamCompaction::Efficient::optimizedScan(testSize, c, a);
        avgOptWorkEfficientTime += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }

    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, c);
        StreamCompaction::Efficient::optimizedMemScan(testSize, c, a);
        avgOptSharedWorkEfficientTime += StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation();
    }

    for (int i = 0; i < trials; i++)
    {
        zeroArray(SIZE, c);
        StreamCompaction::Thrust::scan(testSize, c, a);
        avgThrustTime += StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation();
    }

    printf("==== %s %d, # of trials: %d====\n", "test array size: ", SIZE, trials);

    printDesc("cpu scan, power-of-two");
    std::cout << "   average elapsed time: " << avgCPUTime / trials << "ms    " << "(std::chrono Measured)" << std::endl;
    
    printDesc("naive scan, power-of-two");
    std::cout << "   average elapsed time: " << avgNaiveTime / trials << "ms    " << "(CUDA Measured)" << std::endl;
    
    printDesc("work-efficient scan, power-of-two");
    std::cout << "   average elapsed time: " << avgWorkEfficientTime / trials << "ms    " << "(CUDA Measured)" << std::endl;

    printDesc("optimized work-efficient scan, power-of-two");
    std::cout << "   average elapsed time: " << avgOptWorkEfficientTime / trials << "ms    " << "(CUDA Measured)" << std::endl;

    printDesc("optimized shared mem work-efficient scan, power-of-two");
    std::cout << "   average elapsed time: " << avgOptSharedWorkEfficientTime / trials << "ms    " << "(CUDA Measured)" << std::endl;

    printDesc("thrust scan, power-of-two");
    std::cout << "   average elapsed time: " << avgThrustTime / trials << "ms    " << "(CUDA Measured)" << std::endl;
}

// UNUSED
static void legacyTestingSuite()
{
    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
#if PROFILING
    getAvgTest(50);
#endif

#if !PROFILING
#if !DISABLE_CPU
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

#if ENABLE_NON_POWER_OF_TWO
    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
#endif

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if ENABLE_NON_POWER_OF_TWO
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif

#if USE_OPTIMIZED
    zeroArray(SIZE, c);
    printDesc("optimized work-efficient scan, power-of-two");
    StreamCompaction::Efficient::optimizedScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
       
    zeroArray(SIZE, c);
    printDesc("optimized work-efficient SHARED MEMORY scan, power-of-two");
    StreamCompaction::Efficient::optimizedMemScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if ENABLE_NON_POWER_OF_TWO
    zeroArray(SIZE, c);
    printDesc("optimized work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::optimizedScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif
#endif

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

#if ENABLE_NON_POWER_OF_TWO
    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
#endif

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    //printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    //printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    #endif // PROFILING
}