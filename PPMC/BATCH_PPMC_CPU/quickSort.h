#ifndef QUICKSORT_PARALLEL_H
#define QUICKSORT_PARALLEL_H
#include "stdlib.h"
#include "omp.h"

// 随机创建数组
void rands(int* data, int sum);
// 交换函数
void sw(int* a, int* b);
// 求2的n次幂
int exp2(int wht_num);
// 求log2(n)
int log2(int wht_num);
// 合并两个有序的数组
void mergeList(int* c, int* a, int sta1, int end1, int* b, int sta2, int end2);
// 串行快速排序
int partition(int* a, int sta, int end);
void quickSort(int* a, int sta, int end);
// openMP(8)并行快速排序
void quickSort_parallel(int* array, int lenArray, int numThreads);
void quickSort_parallel_internal(int* array, int left, int right, int cutoff);
#endif