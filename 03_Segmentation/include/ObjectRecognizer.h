#pragma once

#include <cmath>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "my_utils.h"
#include "RealCam.h"

struct ObjectRecognizer
{
private:
    static const size_t W = RealCam::W;
    static const size_t H = RealCam::H;

    static double EPS() { static double val = 1e-5; return val; }
    static const int INF = 1e9;

    static int dx(size_t i) { static int val[4] = {-1,  0, 0, 1}; return val[i]; }
    static int dy(size_t i) { static int val[4] = { 0, -1, 1, 0}; return val[i]; }

    static const size_t PLANE_ITR             = 200;
    static double NORMALS_MAX_DIST()        { static double val = 5;                        return val; } // in mm
    static double PLANE_BOARD_DIST_LOWER()  { static double val = 5;                        return val; } // in pixel
    static double PLANE_BOARD_DIST_UPPER()  { static double val = 10;                       return val; } // in pixel
    static double PLANE_BOARD_ANGLE_LOWER() { static double val = cos(M_PI / 180.0 * 15.0); return val; }
    static double PLANE_BOARD_ANGLE_UPPER() { static double val = cos(M_PI / 180.0 * 50.0); return val; }
    static const size_t FILTER_SIZE           = 200; // in pixel
    static const size_t FAN_DEPTH_PLANE       = 2;   // in pixel
    static const size_t FAN_COLOR_PLANE       = 1;   // in pixel
    static const size_t COMPRESS_DIST         = 10;  // in pixel
    static const size_t COMPRESS_MIN          = 20;  // in pixel
    static const size_t BILATERAL_SIZE        = 19;  // in pixel
    static const size_t OBJECTS_ITR           = 200; //
    static const size_t OBJECTS_PART          = 10;  // in percent %

    struct Plane
    {
    private:
        Eigen::Vector3d _normal;
        double _D;

    public:
        Plane()
        {
            _normal = Eigen::Vector3d::Zero();
            _D = 0;
        }
        Plane(Eigen::Vector3d p1, Eigen::Vector3d p2, Eigen::Vector3d p3)
        {
            Eigen::Vector3d v1 = p1 - p2;
            Eigen::Vector3d v2 = p3 - p2;
            _normal = v1.cross(v2).normalized();
            _D = -_normal.dot(p1);
        }

        void set_normal(Eigen::Vector3d n)
        {
            _normal = n.normalized();
        }

        void set_D(double D)
        {
            _D = D;
        }

        double dist(Eigen::Vector3d v) const
        {
            return abs(_normal.dot(v) + _D);
        }

        double cos_angle(Eigen::Vector3d n) const
        {
            return n.normalized().dot(_normal);
        }
    };

    struct Point_comparator
    {
        bool operator()(const cv::Point& a, const cv::Point& b) const
        {
            if (a.x == b.x) return a.y < b.y;
            return a.x < b.x;
        }
    };

    static Field<Eigen::Vector3d> make_cloud(const RealCam& cam)
    {
        Field<Eigen::Vector3d> cloud(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            cloud(x, y) = cam.depth(x, y).world;
        return cloud;
    }

    static Field<Eigen::Vector3d> calc_normals(const Field<Eigen::Vector3d>& cloud)
    {
        Field<Eigen::Vector3d> normals(W, H);
        for (size_t x = 1; x < W-1; x++)
        for (size_t y = 1; y < H-1; y++)
        {
            if (cloud(x, y).norm() < EPS()) continue;
            if (cloud(x-1, y).norm() < EPS()) continue;
            if (cloud(x+1, y).norm() < EPS()) continue;
            if (cloud(x, y-1).norm() < EPS()) continue;
            if (cloud(x, y+1).norm() < EPS()) continue;
            
            Eigen::Vector3d v_u = cloud(x+1, y) - cloud(x-1, y);
            Eigen::Vector3d v_r = cloud(x, y+1) - cloud(x, y-1);

            if (v_u.norm() > NORMALS_MAX_DIST()) continue;
            if (v_r.norm() > NORMALS_MAX_DIST()) continue;

            normals(x, y) = v_r.cross(v_u).normalized();
        }
        return normals;
    }

    static bool check_point_in_plane(const Plane& plane, Eigen::Vector3d p, Eigen::Vector3d normal, 
                                     const double board_dist = PLANE_BOARD_DIST_LOWER(), const double board_angle = PLANE_BOARD_ANGLE_LOWER())
    {
        return plane.dist(p) < board_dist && plane.cos_angle(normal) > board_angle;
    }

    static Field<bool> get_mask_by_plane(const std::vector<cv::Point>& good_points, const Field<Eigen::Vector3d>& cloud, const Field<Eigen::Vector3d>& normals,
                                         const Plane plane,
                                         const double board_dist, const double board_angle, const bool type)
    {
        Field<bool> mask(W, H);
        for (cv::Point pix : good_points)
        {
            Eigen::Vector3d p = cloud(pix);
            Eigen::Vector3d n = normals(pix);
            if (!check_point_in_plane(plane, p, n, board_dist, board_angle) == type) continue;

            mask(pix) = true;
        }

        return mask;
    }

    static void convert_mask2color(Field<bool>& mask, const RealCam& cam)
    {
        Field<bool> mask_new(W, H);

        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (!mask(x, y)) continue;
            if (!cam.depth(x, y).connect) continue;
            mask_new(cam.depth(x, y).pixel) = true;
        }
        mask = mask_new;
    }

    static void companents_filter(Field<bool>& mask, const bool val)
    {
        Field<int> cmp(W, H);
        cmp.fill(-1);
        size_t num = 0;

        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (mask(x, y) != val) continue;
            if (cmp(x, y) != -1) continue;

            std::vector<cv::Point> Q = {cv::Point(x, y)};
            cmp(x, y) = num;
            for (size_t i = 0; i < Q.size(); i++)
            {
                cv::Point cur = Q[i];

                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                    if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                    if (mask(t) != val) continue;
                    if (cmp(t) != -1) continue;

                    Q.push_back(t);
                    cmp(t) = num;
                }
            }

            num++;
            if (Q.size() < FILTER_SIZE)
            {
                for (cv::Point p : Q)
                    mask(p) = !val;
            }
        }
    }

    static void companents_fan(Field<bool>& mask, size_t d, const bool fan_type)
    {
        std::vector<cv::Point> Q;
        Field<int> dist(W, H);
        dist.fill(-1);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (mask(x, y) != fan_type) continue;
        
            Q.push_back(cv::Point(x, y));
            dist(x, y) = 0;
        }

        for (size_t i = 0; i < Q.size(); i++)
        {
            cv::Point cur = Q[i];

            if (dist(cur) == static_cast<int>(d)) break;

            for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                cv::Point t = cv::Point(cur.x + dx, cur.y + dy);

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (dist(t) != -1) continue;

                Q.push_back(t);
                dist(t) = dist(cur) + 1;
                mask(t) = fan_type;
            }
        }
    }

    static Field<int> find_all_companents(const Field<bool>& mask, const bool val = true)
    {
        Field<int> cmp(W, H);
        cmp.fill(-1);

        size_t num = 0;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (mask(x, y) != val) continue;
            if (cmp(x, y) != -1) continue;

            std::vector<cv::Point> Q = {cv::Point(x, y)};
            cmp(x, y) = num;
            for (size_t i = 0; i < Q.size(); i++)
            {
                cv::Point cur = Q[i];

                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                    if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                    if (mask(t) != val) continue;
                    if (cmp(t) != -1) continue;

                    Q.push_back(t);
                    cmp(t) = num;
                }
            }
            num++;
        }

        return cmp;
    }

    template<class T>
    static bool check_warring_neighbors(cv::Point p, const Field<T>& mask, const T val)
    {
        static const cv::Point dd[8] = {cv::Point(-1, -1), cv::Point(-1,  0), cv::Point(-1,  1), cv::Point( 0,  1), 
                                        cv::Point( 1,  1), cv::Point( 1,  0), cv::Point( 1, -1), cv::Point( 0, -1)};

        size_t cnt = 0;
        bool state = false;
        for (int i = 0; i < 8; i++)
        {
            cv::Point t = cv::Point(p.x + dd[i].x, p.y + dd[i].y);

            if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) 
            {
                state = false;
                continue;
            }

            if (mask(t) == val)
            {
                if (!state)
                {
                    cnt++;
                    state = true;
                }
            }
            else
            {
                state = false;
            }
        }

        if (cnt <= 1) return false;
        if (cnt >  2) return true;

        cv::Point t1 = cv::Point(p.x + dd[0].x, p.y + dd[0].y);
        cv::Point t2 = cv::Point(p.x + dd[7].x, p.y + dd[7].y);

        if (t1.x < 0 || t1.x >= static_cast<int>(W) || t1.y < 0 || t1.y >= static_cast<int>(H)) return true;
        if (t2.x < 0 || t2.x >= static_cast<int>(W) || t2.y < 0 || t2.y >= static_cast<int>(H)) return true;

        return !(mask(t1) == val && mask(t2) == val);
    }

    static Field<int> components_compress(const Field<bool>& mask)
    {
        Field<int> cmp = find_all_companents(mask);

        std::unordered_map<size_t, size_t> cnt;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            if (cmp(x, y) != -1)
                cnt[cmp(x, y)]++;

        std::vector<cv::Point> Q;
        Field<int> dist(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            if (!mask(x, y))
            {
                Q.push_back(cv::Point(x, y));
                dist(x, y) = 0;
            }

        for (size_t i = 0; i < Q.size(); i++)
        {
            cv::Point cur = Q[i];
            
            if (dist(cur) >= static_cast<int>(COMPRESS_DIST)) continue;

            for (size_t d = 0; d < 4; d++)
            {
                cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                if (cmp(t) == -1) continue;
                if (cnt[cmp(t)] < COMPRESS_MIN) continue;
                if (check_warring_neighbors(t, cmp, -1)) continue;

                Q.push_back(t);
                dist(t) = dist(cur) + 1;
                cnt[cmp(t)]--;
                cmp(t) = -1;
            }
        }

        return cmp;
    }

    static Field<int> find_components_plane_by_plane(const RealCam& cam, const std::vector<cv::Point>& good_points, 
                                                     const Field<Eigen::Vector3d>& cloud, const Field<Eigen::Vector3d>& normals, const Plane plane)
    {
        Field<bool> mask = get_mask_by_plane(good_points, cloud, normals, plane, PLANE_BOARD_DIST_LOWER(), PLANE_BOARD_ANGLE_LOWER(), true);

        companents_filter(mask, false);
        companents_fan(mask, FAN_DEPTH_PLANE, false);
        companents_filter(mask, true);

        convert_mask2color(mask, cam);
        companents_fan(mask, FAN_COLOR_PLANE, true);

        return components_compress(mask);
    }

    static Field<int> partition_plane_objects(const RealCam& cam)
    {
        Field<Eigen::Vector3d> cloud = make_cloud(cam);
        Field<Eigen::Vector3d> normals = calc_normals(cloud);

        Plane best_plane;
        size_t best_count = 0;

        std::vector<cv::Point> good_points;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (normals(x, y).norm() < EPS()) continue;
            good_points.push_back(cv::Point(x, y));
        }

        if (good_points.size() < 300) return Field<int>(W, H);

        for (size_t itr = 0; itr < PLANE_ITR; itr++)
        {
            for (int i = 0; i < 3; i++) 
                std::swap(good_points[i], good_points[i + rand() % (good_points.size() - i)]);

            Plane plane(cloud(good_points[0]),
                        cloud(good_points[1]),
                        cloud(good_points[2]));

            size_t count = 0;
            for (size_t i = 0; i < good_points.size(); i += 100)
            {
                cv::Point p = good_points[i];
                if (check_point_in_plane(plane, cloud(p), normals(p)))
                    count++;
            }

            if (best_count < count)
            {
                best_count = count;
                best_plane = plane;
            }
        }

        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d B = Eigen::Vector3d::Zero();
        for (cv::Point pix : good_points)
        {
            Eigen::Vector3d p = cloud(pix);
            Eigen::Vector3d n = normals(pix);
            if (!check_point_in_plane(best_plane, p, n)) continue;

            A(0, 0) += p.x() * p.x();
            A(0, 1) += p.x() * p.y();
            A(0, 2) += p.x() * p.z();
            A(1, 1) += p.y() * p.y();
            A(1, 2) += p.y() * p.z();
            A(2, 2) += p.z() * p.z();
            B(0) -= p.x();
            B(1) -= p.y();
            B(2) -= p.z();
        }
        A(1, 0) = A(0, 1);
        A(2, 0) = A(0, 2);
        A(2, 1) = A(1, 2);
        Eigen::Vector3d normal = A.inverse() * B;
        best_plane.set_normal(normal);
        best_plane.set_D(1 / normal.norm());

        Field<int>  plane_cmp = find_components_plane_by_plane(  cam, good_points, cloud, normals, best_plane);
        
        Field<bool> plane_mask(W, H);
        for (cv::Point pix : good_points)
        {
            Eigen::Vector3d p = cloud(pix);
            Eigen::Vector3d n = normals(pix);
            if (!check_point_in_plane(best_plane, p, n, PLANE_BOARD_DIST_LOWER(), PLANE_BOARD_ANGLE_LOWER())) continue;

            plane_mask(pix) = true;
        }
        Field<bool> objects_mask(W, H);
        for (cv::Point pix : good_points)
        {
            Eigen::Vector3d p = cloud(pix);
            Eigen::Vector3d n = normals(pix);
            if (check_point_in_plane(best_plane, p, n, PLANE_BOARD_DIST_UPPER(), PLANE_BOARD_ANGLE_UPPER())) continue;

            objects_mask(pix) = true;
        }
        static cv::Mat img_depth(cv::Size(W, H), CV_8UC3);
        static cv::Mat img_normals(cv::Size(W, H), CV_8UC3);
        cam.getImageDepth(img_depth);
        draw_mask(img_depth, plane_mask, 1, 255);
        draw_mask(img_depth, objects_mask, 0, 255);
        draw_normals(img_normals, normals);
        SingleImage::add(img_depth);
        SingleImage::add(img_normals);

        return plane_cmp;
    }

    static Field<int> find_kernels_by_compress(const Field<int>& plane)
    {
        Field<bool> cmp_plane(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            cmp_plane(x, y) = (plane(x, y) != -1);

        Field<bool> cmp_object = cmp_plane;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (cmp_plane(x, y)) continue;

            std::vector<cv::Point> Q1 = {cv::Point(x, y)};
            cmp_plane(x, y) = true;
            for (size_t i = 0; i < Q1.size(); i++)
            {
                cv::Point cur = Q1[i];

                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                    if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                    if (cmp_plane(t)) continue;

                    Q1.push_back(t);
                    cmp_plane(t) = true;
                }
            }

            std::vector<cv::Point> Q2;
            std::set<cv::Point, Point_comparator> usd;
            for (cv::Point p : Q1)
            {
                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(p.x + dx(d), p.y + dy(d));

                    if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                    if (plane(t) == -1) continue;
                    if (usd.count(t)) continue;

                    Q2.push_back(t);
                    usd.insert(t);
                }
            }

            for (size_t i = 0; i < Q2.size() && Q1.size() > Q2.size(); i++) //YES! All right.
            {
                cv::Point cur = Q2[i];

                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                    if (t.x < 0 || t.x >= static_cast<int>(W) || t.y < 0 || t.y >= static_cast<int>(H)) continue;
                    if (cmp_object(t)) continue;
                    if (check_warring_neighbors(t, cmp_object, true)) continue;

                    Q2.push_back(t);
                    cmp_object(t) = true;
                }
            }
        }

        return find_all_companents(cmp_object, false);
    }

    static Field<int> segmentation(const cv::Mat& img, const Field<int>& objects_kernels, const Field<int>& plane_kernels)
    {
        std::vector<std::vector<cv::Point>> approx;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            int num = objects_kernels(x, y);
            if (num == -1) continue;

            if (approx.size() <= static_cast<size_t>(num)) approx.resize(num + 1);
            approx[num].push_back(cv::Point(x, y));
        }

        size_t n = approx.size();
        std::vector<std::map<cv::Point, size_t, Point_comparator>> statistic(n);

        cv::Mat blur;
        cv::bilateralFilter(img, blur, BILATERAL_SIZE, 75, 75);
        //cv::imshow("blur", blur);

        for (size_t itr = 0; itr < OBJECTS_ITR; itr++)
        {
            cv::Mat markers = cv::Mat::zeros(cv::Size(W, H), CV_32SC1);

            for (size_t i = 0; i < n; i++)
            {
                std::vector<cv::Point>& iapprox = approx[i];
                size_t cnt = iapprox.size() * OBJECTS_PART / 100;
                for (size_t j = 0; j < cnt; j++)
                    std::swap(iapprox[j], iapprox[j + rand() % (iapprox.size() - j)]);
                for (size_t j = 0; j < cnt; j++)
                    markers.at<int>(iapprox[j]) = 1 + i;
            }
        
            for (size_t x = 0; x < W; x++)
            for (size_t y = 0; y < H; y++)
                if (plane_kernels(x, y) != -1)
                    markers.at<int>(cv::Point(x, y)) = 1 + n + plane_kernels(x, y);

            cv::watershed(blur, markers);

            for (size_t x = 0; x < W; x++)
            for (size_t y = 0; y < H; y++)
            {
                int num = markers.at<int>(cv::Point(x, y));
                if (num <= 0) continue;
                if (num <= static_cast<int>(n))
                    statistic[num - 1][cv::Point(x, y)]++;
            }

            /*cv::Mat img_segments = cv::Mat::zeros(cv::Size(W, H), CV_8UC3);
            for (size_t x = 0; x < W; x++)
            for (size_t y = 0; y < H; y++)
            {
                if (markers.at<int>(cv::Point(x, y)) <= 0) continue;
                if (markers.at<int>(cv::Point(x, y)) <= static_cast<int>(n))
                {
                    img_segments.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255,0,0);
                }
            }
            imshow("segments", img_segments);
            if (cvWaitKey(1) == 27) exit(0);*/
        }

        Field<int> objects(W, H);
        objects.fill(-1);
        for (size_t i = 0; i < n; i++)
        for (auto p : statistic[i])
            if (p.second == OBJECTS_ITR)
                objects(p.first) = i;

        return objects;
    }

    static void draw_normals(cv::Mat& img, const Field<Eigen::Vector3d>& normals)
    {
        img.setTo(cv::Scalar(0,0,0));
        for (size_t x = 1; x < W-1; x++)
        for (size_t y = 1; y < H-1; y++)
        {
            if (normals(x, y).norm() < EPS()) continue;
            img.at<cv::Vec3b>(cv::Point(x, y)) = cv::Vec3b(255 * (normals(x, y)(0) + 1) / 2,
                                                           255 * (normals(x, y)(1) + 1) / 2, 
                                                           255 * (normals(x, y)(2) + 1) / 2);
        }
    }

    static void draw_mask(cv::Mat& img, const Field<bool>& mask, size_t chennel, const unsigned char val)
    {
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (!mask(x, y)) continue;
            cv::Vec3b color = img.at<cv::Vec3b>(cv::Point(x, y));
            color[chennel] = val;
            img.at<cv::Vec3b>(cv::Point(x, y)) = color;
        }
    }

public:
    ObjectRecognizer() = delete;

    static void get_object(const RealCam& cam)
    {
        cv::Mat img(cv::Size(W, H), CV_8UC3);
        cv::Mat img_objects(cv::Size(W, H), CV_8UC3);

        cam.getImageColor(img);

        Field<int> plane_cmp = partition_plane_objects(cam);
        Field<int> objects_kernels = find_kernels_by_compress(plane_cmp);
        Field<int> objects = segmentation(img, objects_kernels, plane_cmp);

        Field<bool> plane_cmp_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            plane_cmp_mask(x, y) = (plane_cmp(x, y) != -1);

        Field<bool> objects_kernels_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            objects_kernels_mask(x, y) = (objects_kernels(x, y) != -1);

        Field<bool> objects_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            objects_mask(x, y) = (objects(x, y) != -1);

        img_objects.setTo(cv::Scalar(0,0,0));
        draw_mask(img, plane_cmp_mask, 1, 255);
        draw_mask(img, objects_kernels_mask, 0, 255);
        draw_mask(img_objects, objects_mask, 2, 255);

        SingleImage::add(img);
        SingleImage::add(img_objects);
    }
};
