#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>


#include "tensor.h"
#include "MicroUNet.h" // 你刚写的 U-Net 结构

using namespace std;
using namespace cv;

// --- 2. 预处理：OpenCV Mat -> MicroTensor ---
Tensor mat_to_tensor(const Mat &img)
{
    Mat resized;
    resize(img, resized, Size(160, 160));

    resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

    Tensor input({1, 3, 160, 160});

    // HWC -> NCHW & BGR -> RGB
    for (int y = 0; y < 160; ++y)
    {
        for (int x = 0; x < 160; ++x)
        {
            Vec3f pixel = resized.at<Vec3f>(y, x);
            input.at_4d(0, 0, y, x) = pixel[2]; // R
            input.at_4d(0, 1, y, x) = pixel[1]; // G
            input.at_4d(0, 2, y, x) = pixel[0]; // B
        }
    }
    return input;
}


void visualize_manual(const Tensor &output, Mat &original_img)
{
    // 1. 获取指针
    const float *p_raw = output.data.data();
    int H = 160, W = 160;

    // 2. 准备 Mask 缓冲区
    std::vector<uchar> mask_data(H * W);

    // 3. Sigmoid + Threshold
    for (int i = 0; i < H * W; ++i)
    {
        float prob = 1.0f / (1.0f + std::exp(-p_raw[i]));
        mask_data[i] = (prob > 0.5f) ? 255 : 0;
    }

    // 4. Resize
    Mat mask_resized(original_img.size(), CV_8UC1);
    float scale_x = (float)W / original_img.cols;
    float scale_y = (float)H / original_img.rows;

    for (int y = 0; y < original_img.rows; y++)
    {
        for (int x = 0; x < original_img.cols; x++)
        {
            int src_x = (int)(x * scale_x);
            int src_y = (int)(y * scale_y);
            mask_resized.at<uchar>(y, x) = mask_data[src_y * W + src_x];
        }
    }

    // 5.  Blending
    Mat color_mask = Mat::zeros(original_img.size(), original_img.type());
    color_mask.setTo(Scalar(0, 255, 0), mask_resized); // 绿色

    addWeighted(original_img, 0.7, color_mask, 0.3, 0, original_img);
}


int main()
{
    string weight_path = "unet_weights.bin";
    string image_path = "test_car.jpg";

   
    cout << ">>> [Init] Building MicroUNet and loading weights..." << endl;
    UNet model;
    model.load_bin(weight_path); // 确保 load_bin 已经写好

    // 2. 加载图片
    Mat img = imread(image_path);
    if (img.empty())
    {
        cerr << "!!! Error: Image load failed!" << endl;
        return -1;
    }
    cout << ">>> [Init] Image Loaded." << endl;

    // 3. 预处理
    Tensor input = mat_to_tensor(img);

    
    cout << ">>> [Core] Running MicroTensor Forward Pass..." << endl;
    auto start = chrono::high_resolution_clock::now();
    Tensor output = model.forward(input);
    auto end = chrono::high_resolution_clock::now();

    cout << ">>> [Core] Inference Done. Time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count() << "ms" << endl;

 
    visualize_manual(output, img);

    // 6. 弹窗显示
    namedWindow("MicroUNet - Hardcore Edition", WINDOW_NORMAL);
    imshow("MicroUNet - Hardcore Edition", img);

    imwrite("micro_unet_result.jpg", img);
    waitKey(0);

    return 0;
}
