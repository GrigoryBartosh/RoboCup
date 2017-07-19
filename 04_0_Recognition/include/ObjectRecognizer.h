#pragma once

#include <cmath>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen> 
#include "my_utils.h"
#include "RealCam.h"

class ObjectRecognizer
{
private:
    static const std::string CONFIG() { static std::string val = "data/config.txt"; return val; }

    static const size_t W = RealCam::W;
    static const size_t H = RealCam::H;

    static double EPS() { static double val = 1e-5; return val; }
    static const size_t INF = 1e9;

    static int dx(size_t i) { static int val[4] = {-1,  0, 0, 1}; return val[i]; }
    static int dy(size_t i) { static int val[4] = { 0, -1, 1, 0}; return val[i]; }

    static const size_t PLANE_ITR             = 200;
    static double NORMALS_MAX_DIST()  { static double val = 5;                        return val; } // in mm
    static double PLANE_BOARD_DIST()  { static double val = 5;                        return val; } // in pixel
    static double PLANE_BOARD_ANGLE() { static double val = cos(M_PI / 180.0 * 15.0); return val; }
    static const size_t FILTER_SIZE           = 200; // in pixel
    static const size_t FAN_DEPTH_PLANE       = 2;   // in pixel
    static const size_t FAN_COLOR_PLANE       = 1;   // in pixel
    static const size_t COMPRESS_DIST         = 10;  // in pixel
    static const size_t COMPRESS_MIN          = 20;  // in pixel
    static const size_t BILATERAL_SIZE        = 19;  // in pixel
    static const size_t OBJECTS_ITR           = 200; //
    static const size_t OBJECTS_PART          = 10;  // in percent %
    static const size_t ICP_ITR               = 10;  //
    static const size_t ICP_MAX_ITR           = 30;  //
    static const size_t BLACK_BOARD           = 80;  //

    enum ObjectType { F20_20_H = 0, F20_20_HB = 9,  F20_20_HG = 10, F20_20_V = 13,
                      F40_40_H = 1, F40_40_HB = 11, F40_40_HG = 12, F40_40_V = 14,
                      M20_100_H = 2, M20_100_V = 15,
                      M20_H = 3, M20_V = 4,
                      M30_H = 5, M30_V = 6,
                      R20_H = 7, R20_V = 8 };

    struct Plane
    {
    private:
        Eigen::Vector3d _normal;
        double _D;

    public:
        Plane();
        Plane(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3);

        void set_normal(Eigen::Vector3d n);
        void set_D(double D);
        double dist(Eigen::Vector3d v) const;
        double cos_angle(Eigen::Vector3d n) const;
    };

    struct Pattern
    {
    private:
        std::vector<cv::Point> _points;
        Field<cv::Point> _field;
        cv::Point _center;

        std::vector<cv::Point> read(std::string file_name);
        std::vector<cv::Point> get_contours(const std::vector<cv::Point>& points);
        void init_voronoi(const std::vector<cv::Point>& points);
        void init_all(const std::vector<cv::Point>& points);
        Eigen::Affine2d make_transform(double a, double tx, double ty);
        Eigen::Affine2d icp_iteration(const std::vector<cv::Point>& object, const Eigen::Affine2d& T);
        bool isMinorTransformation(const Eigen::Affine2d& T);
        double get_fitness(const std::vector<cv::Point>& object, const Eigen::Affine2d& T);
        void draw(const std::vector<cv::Point>& object, const Eigen::Affine2d T);

    public:
        Pattern() = default;
        Pattern(std::string file_name);

        void init(std::string file_name);

        std::tuple<double, double> compare(const std::vector<cv::Point>& object, const cv::Point center);
    };

    struct Object
    {
        cv::Point pos;
        double angle;
        ObjectType type;
    };

    struct Point_comparator
    {
        bool operator()(const cv::Point& a, const cv::Point& b) const
        {
            if (a.x == b.x) return a.y < b.y;
            return a.x < b.x;
        }
    };

    std::vector<Pattern> _patterns;

    Field<Eigen::Vector3d> make_cloud(const RealCam& cam);
    Field<Eigen::Vector3d> calc_normals(const Field<Eigen::Vector3d>& cloud);

    bool check_point_in_plane(const Plane& plane, Eigen::Vector3d p, Eigen::Vector3d normal);
    Field<bool> get_mask_by_plane(const std::vector<cv::Point>& good_points, const Field<Eigen::Vector3d>& cloud, const Field<Eigen::Vector3d>& normals,
                                  const Plane plane);
    void convert_mask2color(Field<bool>& mask, const RealCam& cam);
    void companents_filter(Field<bool>& mask, const bool val);
    void companents_fan(Field<bool>& mask, size_t d, const bool fan_type);
    Field<int> find_all_companents(const Field<bool>& mask, const bool val = true);
    template<class T>
    bool check_warring_neighbors(cv::Point p, const Field<T>& mask, const T val);
    Field<int> components_compress(const Field<bool>& mask);

    Field<int> find_components_plane_by_plane(const RealCam& cam, const std::vector<cv::Point>& good_points, 
                                                     const Field<Eigen::Vector3d>& cloud, const Field<Eigen::Vector3d>& normals, const Plane plane);

    Field<int> find_plane(const RealCam& cam);
    Field<int> find_kernels_by_compress(const Field<int>& plane);
    Field<int> segmentation(const cv::Mat& img, const Field<int>& objects_kernels, const Field<int>& plane_kernels);
    std::vector<Object> recognition(const Field<int>& objects);
    void clarify_color(std::vector<Object>& types, const Field<int>& objects, const cv::Mat& img);

    void draw_normals(cv::Mat& img, const Field<Eigen::Vector3d>& normals);
    void draw_mask(cv::Mat& img, const Field<bool>& mask, size_t chennel, const unsigned char val);

public:
    ObjectRecognizer();
    void get_object(const RealCam& cam);
};