#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <chrono> // 包含 chrono 库
#include <opencv2/opencv.hpp>

struct ROI {
    int x;
    int y;
    int width;
    int height;
    // Parameterized constructor
    ROI(int x, int y, int width, int height)
        : x(x), y(y), width(width), height(height) {
    }
};

std::vector<ROI> splitImageWithOverlap( int imgWidth,
                                        int imgHeight,
                                        int startX,
                                        int startY,
                                        int roiWidth,
                                        int roiHeight,
                                        int stepX,
                                        int stepY) {
    std::vector<ROI> rois;
    // 计算 ROI 的数量
    for (int y = startY; y + roiHeight <= imgHeight; y += stepY) {
        for (int x = startX; x + roiWidth <= imgWidth; x += stepX) {
            rois.push_back(ROI{x,y,roiWidth ,roiHeight });
        }
    }
    return rois;
}

// CUDA内核：基于多个ROI裁剪图像 (2D Kernel)
__global__ void extractRoiKernel(const unsigned char* d_image, unsigned char* d_output, ROI* d_rois , int roi_count,int width, int height, int channels) {
    // 计算线程的全局索引
    int tx = threadIdx.x + blockIdx.x * blockDim.x; // 横向索引
    int ty = threadIdx.y + blockIdx.y * blockDim.y; // 纵向索引
    // 确保索引在有效范围内
    if (tx < roi_count) {
        // 获取当前 ROI
        ROI roi = d_rois[tx];
        // 计算该 ROI 的位置和大小
        int roi_x = roi.x;
        int roi_y = roi.y;
        int roi_width = roi.width;
        int roi_height = roi.height;
        // 遍历 ROI 区域并提取图像数据
        for (int i = 0; i < roi_height; i++) {
            for (int j = 0; j < roi_width; j++) {
                int global_x = roi_x + j;  // 当前像素的全局 x 坐标
                int global_y = roi_y + i;  // 当前像素的全局 y 坐标
                // 检查该坐标是否在图像范围内
                if (global_x < width && global_y < height) {
                    int image_index = (global_y * width + global_x) * channels;  // 图像的索引
                    int output_index = (tx * roi_width * roi_height + i * roi_width + j) * channels;  // 输出的索引
                    // 提取每个通道的像素值
                    for (int c = 0; c < channels; c++) {
                        d_output[output_index + c] = d_image[image_index + c];
                    }
                }
            }
        }
    }
}


/*
// CUDA 内核：根据多个 ROI 裁剪图像
__global__ void cropImageKernel(const uchar* inputImage, uchar* outputImage, int imageWidth, int imageHeight, int numROIs, ROI* rois) {
    int roiIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (roiIdx < numROIs) {
        ROI roi = rois[roiIdx];

        // 避免越界
        if (roi.x + roi.width <= imageWidth && roi.y + roi.height <= imageHeight) {
            // 计算每个ROI的起始位置和输出图像的偏移量
            int outputOffset = roiIdx * roi.width * roi.height;

            for (int y = 0; y < roi.height; ++y) {
                for (int x = 0; x < roi.width; ++x) {
                    int inputOffset = (roi.y + y) * imageWidth + (roi.x + x);
                    int outputIdx = outputOffset + y * roi.width + x;
                    outputImage[outputIdx] = inputImage[inputOffset];
                }
            }
        }
    }
}

*/

int main() {
    // 读取图像
    cv::Mat image = cv::imread("1.bmp", cv::IMREAD_GRAYSCALE);

    // 检查图像是否成功读取
    if (image.empty()) {
        std::cerr << "无法读取图像!" << std::endl;
        return -1;
    }
    // 获取图像的宽度、高度和通道数
    int img_width = image.cols;   // 图像的宽度
    int img_height = image.rows;  // 图像的高度
    int img_channels = image.channels();  // 图像的通道数，灰度图像通常为 1

    // 重叠区域
    int overlap_x = 28;
    int overlap_y = 128;

    // 子图像尺寸
    int subWidth = 640;
    int subHeight = 640;

    // 步长
    int strideX = subWidth - overlap_x; // 横向步长
    int strideY = subHeight - overlap_y; // 纵向步长

    // 定义多个 ROI 区域
    std::vector<ROI> rois = splitImageWithOverlap(img_width, img_height, 0, 0, subWidth, subHeight, strideX, strideY);
    int numROIs = rois.size();
    // 在 GPU 上分配内存
    uchar* d_inputImage;
    uchar* d_outputImage;
    ROI* d_rois;

    // 分配图像数据内存
    cudaMalloc(&d_inputImage, img_width * img_height * sizeof(uchar));
    cudaMalloc(&d_outputImage, numROIs * 640 * 640 * sizeof(uchar));  // 假设每个 ROI 区域最大为 100x100
    cudaMalloc(&d_rois, numROIs * sizeof(ROI));

    // 将图像数据和 ROI 数据从主机传输到 GPU
    cudaMemcpy(d_inputImage, image.data, img_width * img_height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rois, rois.data(), numROIs * sizeof(ROI), cudaMemcpyHostToDevice);

    // 调用 CUDA 内核进行裁剪
    //int threadsPerBlock = 1024;
    //int blocksPerGrid = (numROIs + threadsPerBlock - 1) / threadsPerBlock;
    //cropImageKernel <<<blocksPerGrid, threadsPerBlock >> > (d_inputImage, d_outputImage, img_width, img_height, numROIs, d_rois);
     // 调用 CUDA 内核进行裁剪
    // 确保每个线程块处理一个ROI
    
    auto start = std::chrono::high_resolution_clock::now();
    dim3 blockSize(32, 16);
    dim3 gridSize((rois.size() + blockSize.x - 1) / blockSize.x, 1);
    // 启动 CUDA 核函数

    for (int i = 0; i < 100; i++) {
        extractRoiKernel <<<gridSize, blockSize >> > (d_inputImage, d_outputImage, d_rois, numROIs, img_width, img_height, img_channels);
        // 等待 CUDA 核函数执行完毕
        cudaDeviceSynchronize();
    }


    // 检查 CUDA 内核执行是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 获取程序结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间差（持续时间）
    std::chrono::duration<double, std::milli> duration = end - start;

    // 输出执行时间（单位：毫秒）
    std::cout << "Execution time: " << duration.count() << " milliseconds" << std::endl;

    // 从 GPU 拷贝裁剪结果回 CPU
    std::vector<uchar> outputImage(numROIs * 640 * 640);       // 假设最大尺寸是 100x100
    cudaMemcpy(outputImage.data(), d_outputImage, numROIs * 640 * 640 * sizeof(uchar), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_rois);

    // 创建裁剪图像并显示（假设裁剪区域的最大尺寸为 100x100）
    for (int i = 0; i < numROIs; ++i) {
        ROI roi = rois[i];
        cv::Mat croppedImage(roi.height, roi.width, CV_8UC1, outputImage.data() + i * roi.width * roi.height);
        cv::imwrite("Cropped Image " + std::to_string(i) + ".bmp", croppedImage);
    }

    return 0;
}
