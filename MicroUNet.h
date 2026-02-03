#pragma once
#include "tensor.h"
class ConvBlock
{
public:
    Tensor weight;
    Tensor bias;
    bool use_act;

    int k, s, p;
    ConvBlock(int c1, int c2, int k, int s, int p = -1, bool act = true)
        : k(k), s(s), use_act(act)
    {
        if (p == -1)
        {
            this->p = k / 2;
        }
        else
        {
            this->p = p;
        }
        weight = Tensor::randn({c2, c1, k, k});
        bias = Tensor::randn({c2});
    }

    void load_weights(std::ifstream &f)
    {
        // 读权重
        f.read((char *)weight.data.data(), weight.data.size() * sizeof(float));
        // 读偏置
        f.read((char *)bias.data.data(), bias.data.size() * sizeof(float));
    }

    Tensor forward(const Tensor &x)
    {
        Tensor out = conv2d(x, weight, bias, s, p);
        if (use_act)
        {
            return out.SiLU();
        }
        return out;
    }
};

// 双卷积层
class DoubleConv
{
public:
    ConvBlock conv1;
    ConvBlock conv2;

    DoubleConv(int in_channels, int out_channels)
        : conv1(in_channels, out_channels, 3, 1, 1, false),
          conv2(out_channels, out_channels, 3, 1, 1, false)
    {
    }

    Tensor forward(const Tensor &x)
    {
        Tensor out = conv1.forward(x);
        out.UNet_ReLU();
        out = conv2.forward(out);
        out.UNet_ReLU();
        return out;
    }
};

// UNet网络结构
class UNet
{
public:
    std::vector<DoubleConv> downs;
    DoubleConv bottleneck;
    std::vector<DoubleConv> ups;
    ConvBlock final_conv;

    UNet(int in_channels = 3, int out_channels = 1)
        : downs({DoubleConv(in_channels, 64),
                 DoubleConv(64, 128),
                 DoubleConv(128, 256),
                 DoubleConv(256, 512)}),
          bottleneck(512, 1024),
          ups({DoubleConv(1536, 512),
               DoubleConv(768, 256),
               DoubleConv(384, 128),
               DoubleConv(192, 64)}),
          final_conv(64, out_channels, 1, 1, 0, false)
    {
    }

    void load_bin(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
        {
            std::cerr << "Error: Cannot open weights file: " << path << std::endl;
            exit(1);
        }

        auto read_conv = [&](ConvBlock &cb)
        {
            f.read(reinterpret_cast<char *>(cb.weight.data.data()), cb.weight.data.size() * sizeof(float));
            f.read(reinterpret_cast<char *>(cb.bias.data.data()), cb.bias.data.size() * sizeof(float));
        };

        auto read_double = [&](DoubleConv &dc)
        {
            read_conv(dc.conv1);
            read_conv(dc.conv2);
        };

        for (size_t i = 0; i < downs.size(); ++i)
            read_double(downs[i]);

        read_double(bottleneck);

        for (size_t i = 0; i < ups.size(); ++i)
            read_double(ups[i]);

        read_conv(final_conv);
        f.close();
    }

    Tensor forward(const Tensor &x)
    {
        Tensor x1 = downs[0].forward(x);
        Tensor p1 = x1.MaxPool2x2();

        Tensor x2 = downs[1].forward(p1);
        Tensor p2 = x2.MaxPool2x2();

        Tensor x3 = downs[2].forward(p2);
        Tensor p3 = x3.MaxPool2x2();

        Tensor x4 = downs[3].forward(p3);
        Tensor p4 = x4.MaxPool2x2();

        Tensor bottleneck_out = bottleneck.forward(p4);

        Tensor u1_up = bottleneck_out.Upsample2x();
        Tensor u1_cat = Tensor::Concat({x4, u1_up}, 1);
        Tensor u1 = ups[0].forward(u1_cat);

        Tensor u2_up = u1.Upsample2x();
        Tensor u2_cat = Tensor::Concat({x3, u2_up}, 1);
        Tensor u2 = ups[1].forward(u2_cat);

        Tensor u3_up = u2.Upsample2x();
        Tensor u3_cat = Tensor::Concat({x2, u3_up}, 1);
        Tensor u3 = ups[2].forward(u3_cat);

        Tensor u4_up = u3.Upsample2x();
        Tensor u4_cat = Tensor::Concat({x1, u4_up}, 1);
        Tensor u4 = ups[3].forward(u4_cat);

        return final_conv.forward(u4);
    }
};