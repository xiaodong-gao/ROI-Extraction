#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>

struct ROI {
    int x, y, width, height;
};

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

int main() {
    // 读取图像
    cv::Mat image = cv::imread("image.bmp", cv::IMREAD_UNCHANGED);

    // 检查图像是否成功读取
    if (image.empty()) {
        std::cerr << "无法读取图像!" << std::endl;
        return -1;
    }
    // 获取图像的宽度、高度和通道数
    int width = image.cols;   // 图像的宽度
    int height = image.rows;  // 图像的高度
    int channels = image.channels();  // 图像的通道数，灰度图像通常为 1

    int cellWidth = width / 8;  // 计算每个单元格的宽度
    int cellHeight = height / 8; // 计算每个单元格的高度

    std::vector<ROI> rois;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            ROI roi;
            roi.x = j * cellWidth;
            roi.y = i * cellHeight;
            roi.width = cellWidth;
            roi.height = cellHeight;
            rois.push_back(roi);

        }
    }

    int numROIs = rois.size();

    // 在 GPU 上分配内存
    uchar* d_inputImage;
    uchar* d_outputImage;
    ROI* d_rois;

    // 分配图像数据内存
    cudaMalloc(&d_inputImage, width * height * sizeof(uchar));
    cudaMalloc(&d_outputImage, numROIs * 640 * 640 * sizeof(uchar));  // 假设每个 ROI 区域最大为 100x100
    cudaMalloc(&d_rois, numROIs * sizeof(ROI));

    // 将图像数据和 ROI 数据从主机传输到 GPU
    cudaMemcpy(d_inputImage, image.data, width * height * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rois, rois.data(), numROIs * sizeof(ROI), cudaMemcpyHostToDevice);

    // 调用 CUDA 内核进行裁剪
    int threadsPerBlock = 256;
    int blocksPerGrid = (numROIs + threadsPerBlock - 1) / threadsPerBlock;
    cropImageKernel <<<blocksPerGrid, threadsPerBlock >> > (d_inputImage, d_outputImage, width, height, numROIs, d_rois);

    // 检查 CUDA 内核执行是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 从 GPU 拷贝裁剪结果回 CPU
    std::vector<uchar> outputImage(numROIs * 640 * 640);  // 假设最大尺寸是 100x100
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
