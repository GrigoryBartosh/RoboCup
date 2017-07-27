#pragma once

#include <cstdio>
#include <cassert>
#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "my_utils.h"

struct Point4d
{
    unsigned char r, g, b;
    unsigned int d;
    bool connect;
    cv::Point pixel;
    Eigen::Vector3d world;
    Point4d()
    {
        r = 0;
        g = 0;
        b = 0;
        d = 0;
        connect = false;
        pixel = cv::Point(-1, -1);
        world = Eigen::Vector3d::Zero();
    }
};

class RealCam
{
private:
    const unsigned int DIST_MIN = 500;
    const unsigned int DIST_MAX = 10000;

    rs::context _ctx;
    rs::device* _dev;
    std::vector<Point4d> _color;
    std::vector<Point4d> _depth; 

    void drawDepth(cv::Mat& img, const std::vector<Point4d> &arr) const;
    Eigen::Vector3d to_vector3d(rs::float3 f) const;

public:
    static const size_t W = 640;
    static const size_t H = 480;
    
    RealCam();
    ~RealCam();
    RealCam(const RealCam& other) = delete;
    RealCam& operator=(const RealCam& other) = delete;

    void update();

    Point4d color(size_t x, size_t y) const;
    Point4d depth(size_t x, size_t y) const;

    void getImageColor(cv::Mat& img)            const;
    void getImageDepth(cv::Mat& img)            const;
    void getImageColor_mask(cv::Mat& img)       const;
    void getImageDepth_translated(cv::Mat& img) const;

    Eigen::Vector3d deproject(const cv::Point p) const;
};