#include "freshman.h"
#include <chrono>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>

int main() {
    // 设置向量大小
    const int size = 100000000;

    // 创建一个包含大量随机整数的设备向量
    thrust::device_vector<int> d_vec(size);

    // 使用Thrust生成随机整数
    thrust::counting_iterator<int> count_begin(0);
    thrust::generate(thrust::device, d_vec.begin(), d_vec.end(), [=] __device__() mutable {
        thrust::default_random_engine rng;
        thrust::uniform_int_distribution<int> dist(1, 1000);
        return dist(rng);
    });

    // 记录排序前的时间
    // auto start = std::chrono::high_resolution_clock::now();
    double iStart, iElaps;
    iStart = cpuSecond();

    // 使用thrust::sort对向量进行排序
    thrust::sort(thrust::device, d_vec.begin(), d_vec.end());
    iElaps = cpuSecond() - iStart;
    // 记录排序后的时间
    // auto end = std::chrono::high_resolution_clock::now();

    // 计算排序所需的时间
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 打印排序时间
    std::cout << "Sorting took " << iElaps * 1000 << " milliseconds." << std::endl;

    return 0;
}

// #include <iostream>
// #include <vector>
// #include <algorithm>
// #include <chrono>

// int main() {
//     const int size = 1000000;

//     // Create a vector with random integers
//     std::vector<int> data(size);
//     std::generate(data.begin(), data.end(), []() { return rand() % 1000; });

//     // Record the start time
//     auto start = std::chrono::high_resolution_clock::now();

//     // Sort the vector
//     std::sort(data.begin(), data.end());

//     // Record the end time
//     auto end = std::chrono::high_resolution_clock::now();

//     // Calculate the duration
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//     // Print the duration
//     std::cout << "Sorting took " << duration << " milliseconds." << std::endl;

//     return 0;
// }
