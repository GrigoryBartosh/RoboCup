#include "RealCam.h"

using std::vector;
using Eigen::Vector3d;

using std::round;
using std::min;
using std::max;

void RealCam::drawDepth(cv::Mat& img, const vector<Point4d> &arr) const
{
    assert(img.data && static_cast<size_t>(img.cols) == W && static_cast<size_t>(img.rows) == H && static_cast<size_t>(img.channels()) >= 1 && "invalid image");

    for (size_t y = 0; y < H; y++)
    for (size_t x = 0; x < W; x++)
    {
        int d = arr[W * y + x].d;
        unsigned char color = 255 * (d - DIST_MIN + 1) / (DIST_MAX - DIST_MIN + 1);
        if (d == 0) color = 0;

        if (img.type() == CV_8UC3) img.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(color,color,color);
        else                       img.at<unsigned char>(cv::Point(x, y)) = color;
    }
}

Vector3d RealCam::to_vector3d(rs::float3 f) const
{
    Vector3d v;
    v << f.x * 1000.0, f.y * 1000.0, f.z * 1000.0;
    return v;
}

RealCam::RealCam()
{
    _dev = _ctx.get_device(0);

    _dev->enable_stream(rs::stream::depth, rs::preset::best_quality);
    _dev->enable_stream(rs::stream::color, rs::preset::best_quality);
    _dev->start();

    _color.resize(W * H);
    _depth.resize(W * H);
}

RealCam::~RealCam()
{

}

void RealCam::update()
{
    _dev->wait_for_frames();

    const uint16_t* depth_image = (const uint16_t*)_dev->get_frame_data(rs::stream::depth);
    const uint8_t*  color_image = (const uint8_t* )_dev->get_frame_data(rs::stream::color);
    
    rs::intrinsics depth_intrin = _dev->get_stream_intrinsics(rs::stream::depth);
    rs::extrinsics depth_to_color = _dev->get_extrinsics(rs::stream::depth, rs::stream::color);
    rs::intrinsics color_intrin = _dev->get_stream_intrinsics(rs::stream::color);
    float scale = _dev->get_depth_scale();

    _color.assign(W * H, Point4d());
    _depth.assign(W * H, Point4d());
    for (size_t y = 0; y < H; y++)
    for (size_t x = 0; x < W; x++)
    {
        uint16_t depth_value = depth_image[W * y + x];
        float depth_in_meters = depth_value * scale;

        size_t pixel_num_d = W * y + x;
        _depth[pixel_num_d].d = depth_value;

        if (depth_value < DIST_MIN || depth_value > DIST_MAX) continue;
        
        rs::float2 depth_pixel = {(float)x, (float)y};
        rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
        rs::float3 color_point = depth_to_color.transform(depth_point);
        rs::float2 color_pixel = color_intrin.project(color_point);

        _depth[pixel_num_d].world = to_vector3d(depth_point);

        const int cx = (int)round(color_pixel.x), cy = (int)round(color_pixel.y);

        if(cx < 0 || cy < 0 || cx >= static_cast<int>(W) || cy >= static_cast<int>(H)) continue;
    
        size_t pixel_num_c = W * cy + cx;
        _color[pixel_num_c].d = depth_value;
        _depth[pixel_num_d].r = color_image[pixel_num_c * 3];
        _depth[pixel_num_d].g = color_image[pixel_num_c * 3 + 1];
        _depth[pixel_num_d].b = color_image[pixel_num_c * 3 + 2];
        _color[pixel_num_c].connect = true;
        _depth[pixel_num_d].connect = true;
        _color[pixel_num_c].pixel = cvPoint( x,  y);
        _depth[pixel_num_d].pixel = cvPoint(cx, cy);
        _color[pixel_num_c].world = to_vector3d(depth_point);
    }

    for (size_t y = 0; y < H; y++)
    for (size_t x = 0; x < W; x++)
    {
        size_t pixel_num = W * y + x;
        _color[pixel_num].r = color_image[pixel_num * 3];
        _color[pixel_num].g = color_image[pixel_num * 3 + 1];
        _color[pixel_num].b = color_image[pixel_num * 3 + 2];
    }
}

Point4d RealCam::color(size_t x, size_t y) const
{
    assert(x < W && y < H && "Ivalid coordinates");
    return _color[W * y + x];
}

Point4d RealCam::depth(size_t x, size_t y) const
{
    assert(x < W && y < H && "Ivalid coordinates");
    return _depth[W * y + x];
}

void RealCam::getImageColor(cv::Mat& img) const
{
    assert(img.data && static_cast<size_t>(img.cols) == W && static_cast<size_t>(img.rows) == H && img.type() == CV_8UC3 && "invalid image");

    for (size_t y = 0; y < H; y++)
    for (size_t x = 0; x < W; x++)
        img.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(color(x, y).b, color(x, y).g, color(x, y).r);
}

void RealCam::getImageDepth(cv::Mat& img) const
{
    drawDepth(img, _depth);
}

void RealCam::getImageColor_mask(cv::Mat& img) const
{
    assert(img.data && static_cast<size_t>(img.cols) == W && static_cast<size_t>(img.rows) == H && img.type() == CV_8UC3 && "invalid image");

    getImageColor(img);
    for (size_t y = 0; y < H; y++)
    for (size_t x = 0; x < W; x++)
        if (!color(x, y).connect)
        img.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(0, 0, 0);
}   

void RealCam::getImageDepth_translated(cv::Mat& img) const
{
    drawDepth(img, _color);
}