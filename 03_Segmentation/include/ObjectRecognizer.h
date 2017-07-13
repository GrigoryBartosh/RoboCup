#pragma once

#include <cmath> 
#include <ctime> 
#include <algorithm>
#include <set>
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

    static const size_t PLANE_ITR = 200;
    static double PLANE_BOARD_DIST()  { static double val = 5;                        return val; }
    static double PLANE_BOARD_ANGLE() { static double val = cos(M_PI / 180.0 * 15.0); return val; }
    static const size_t FAN_DEPTH      = 2;   // in pixel
    static const size_t FAN_COLOR      = 1;   // in pixel
    static const size_t FILTER_SIZE    = 200; // in pixel
    static const size_t HIST_PERCENT   = 95;  // from 0 to 100
    static const size_t PLANE_COMPRESS = 10;  // in pixel
    static const size_t PLANE_MIN      = 20;  // in pixel

    //TODO static_cast

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

            normals(x, y) = v_r.cross(v_u).normalized();
        }
        return normals;
    }

    static bool check_point(const Plane& plane, Eigen::Vector3d p, Eigen::Vector3d normal)
    {
        return plane.dist(p) < PLANE_BOARD_DIST() && plane.cos_angle(normal) > PLANE_BOARD_ANGLE();
    }

    static void plane_filter(Field<bool>& mask, const bool val)
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

    static void plane_fan(Field<bool>& mask, size_t d, const bool fan_type)
    {
        std::vector<cv::Point> Q;
        Field<int> dist(W, H);
        dist.fill(-1);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            if (mask(x, y) == fan_type)
            {
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

    static Field<bool> find_plane(const RealCam& cam)
    {
        static cv::Mat img_depth(cv::Size(W, H), CV_8UC3);
        static cv::Mat img_normals(cv::Size(W, H), CV_8UC3);

        cam.getImageDepth(img_depth);

        Field<Eigen::Vector3d> cloud = make_cloud(cam);
        Field<Eigen::Vector3d> normals = calc_normals(cloud);

        Field<bool> plane_mask(W, H);
        Plane best_plane;
        size_t best_count = 0;

        std::vector<cv::Point> good_points;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (normals(x, y).norm() < EPS()) continue;
            good_points.push_back(cv::Point(x, y));
        }
        random_shuffle(good_points.begin(), good_points.end());

        if (good_points.size() < 3) return plane_mask;

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
                if (check_point(plane, cloud(p), normals(p)))
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
            if (!check_point(best_plane, p, n)) continue;

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

        for (cv::Point pix : good_points)
        {
            Eigen::Vector3d p = cloud(pix);
            Eigen::Vector3d n = normals(pix);
            if (!check_point(best_plane, p, n)) continue;

            plane_mask(pix) = true;
        }

        plane_filter(plane_mask, false);
        plane_fan(plane_mask, FAN_DEPTH, false);
        plane_filter(plane_mask, true);

        draw_normals(img_normals, normals);
        draw_mask(img_depth, plane_mask, 1, 255);

        convert_mask2color(plane_mask, cam);
        plane_fan(plane_mask, FAN_COLOR, true);

        SingleImage::add(img_depth);
        SingleImage::add(img_normals);

        return plane_mask;
    }

    static Field<bool> find_objects_by_hist(const cv::Mat& img, const Field<bool>& plane)
    {
        cv::Scalar board_down, board_up;

        std::tie(board_down, board_up) = boards_by_hist(img, plane, HIST_PERCENT);

        Field<bool> objects(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (check_color_in_boards(img.at<cv::Vec3b>(cv::Point(x, y)), board_down, board_up)) continue;

            objects(x, y) = true;
        }

        return objects;
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

            if (t.x < 0 || t.x >= (int)W || t.y < 0 || t.y >= (int)H) 
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

        if (t1.x < 0 || t1.x >= (int)W || t1.y < 0 || t1.y >= (int)H) return true;
        if (t2.x < 0 || t2.x >= (int)W || t2.y < 0 || t2.y >= (int)H) return true;

        return !(mask(t1) == val && mask(t2) == val);
    }

    static Field<int> find_objects_by_compress(const Field<bool>& plane)
    {
        struct comparator_cvPoint
        {
            bool operator()(const cv::Point& a, const cv::Point& b) const
            {
                if (a.x == b.x) return a.y < b.y;
                return a.x < b.x;
            }
        };

        Field<int> cmp_plane_int = find_all_companents(plane);
        Field<bool> cmp_plane(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            cmp_plane(x, y) = (cmp_plane_int(x, y) != -1);

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

                    if (t.x < 0 || t.x >= (int)W || t.y < 0 || t.y >= (int)H) continue;
                    if (cmp_plane(t)) continue;

                    Q1.push_back(t);
                    cmp_plane(t) = true;
                }
            }

            std::vector<cv::Point> Q2;
            std::set<cv::Point, comparator_cvPoint> usd;
            for (cv::Point p : Q1)
            {
                for (size_t d = 0; d < 4; d++)
                {
                    cv::Point t = cv::Point(p.x + dx(d), p.y + dy(d));

                    if (t.x < 0 || t.x >= (int)W || t.y < 0 || t.y >= (int)H) continue;
                    if (!plane(t)) continue;
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

                    if (t.x < 0 || t.x >= (int)W || t.y < 0 || t.y >= (int)H) continue;
                    if (cmp_object(t)) continue;
                    if (check_warring_neighbors(t, cmp_object, true)) continue;

                    Q2.push_back(t);
                    cmp_object(t) = true;
                }
            }
        }

        return find_all_companents(cmp_object, false);
    }

    static Field<int> compress_plane(const Field<bool>& mask)
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
            
            if (dist(cur) >= static_cast<int>(PLANE_COMPRESS)) continue;

            for (size_t d = 0; d < 4; d++)
            {
                cv::Point t = cv::Point(cur.x + dx(d), cur.y + dy(d));

                if (t.x < 0 || t.x >= (int)W || t.y < 0 || t.y >= (int)H) continue;
                if (cmp(t) == -1) continue;
                if (cnt[cmp(t)] < PLANE_MIN) continue;
                if (check_warring_neighbors(t, cmp, -1)) continue;

                Q.push_back(t);
                dist(t) = dist(cur) + 1;
                cnt[cmp(t)]--;
                cmp(t) = -1;
            }
        }

        return cmp;
    }

    static Field<int> segmentation(cv::Mat& img, const Field<int>& object_kernels, const Field<int>& plane_kernels)
    {
        cv::Mat markers = cv::Mat::zeros(cv::Size(W, H), CV_32SC1);

        size_t sz = 0;
        std::map<int, int> mp;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (object_kernels(x, y) == -1) continue;
            markers.at<int>(cv::Point(x, y)) = 1 + object_kernels(x, y);
            sz = std::max(static_cast<int>(sz), 1 + object_kernels(x, y));
        }
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (plane_kernels(x, y) == -1) continue;
            markers.at<int>(cv::Point(x, y)) = 1 + sz + plane_kernels(x, y);
        }
        
        cv::watershed(img, markers);

        Field<int> objects(W, H);
        objects.fill(-1);
        mp.clear();
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (markers.at<int>(cv::Point(x, y)) == -1) continue;
            if (markers.at<int>(cv::Point(x, y)) <= static_cast<int>(sz))
                objects(x, y) = markers.at<int>(cv::Point(x, y)) - 1;
        }

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

    static void draw_mask(cv::Mat& img, const Field<bool>& mask, size_t chennel, unsigned char val)
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
        static cv::Mat img(cv::Size(W, H), CV_8UC3);
        static cv::Mat img_objects(cv::Size(W, H), CV_8UC3);

        cam.getImageColor(img);

        Field<bool> plane = find_plane(cam);
        //Field<bool> object_points = find_objects_by_hist(img, plane);
        Field<int> object_kernels = find_objects_by_compress(plane);
        Field<int> plane_cmp = compress_plane(plane);
        Field<int> objects = segmentation(img, object_kernels, plane_cmp);

        Field<bool> object_kernels_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            object_kernels_mask(x, y) = (object_kernels(x, y) != -1);

        Field<bool> plane_cmp_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            plane_cmp_mask(x, y) = (plane_cmp(x, y) != -1);

        Field<bool> objects_mask(W, H);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
            objects_mask(x, y) = (objects(x, y) != -1);

        img_objects.setTo(cv::Scalar(0,0,0));
        draw_mask(img, object_kernels_mask, 0, 255);
        draw_mask(img, plane_cmp_mask, 1, 255);
        draw_mask(img, objects_mask, 2, 255);
        draw_mask(img_objects, objects_mask, 2, 255);

        SingleImage::add(img);
        SingleImage::add(img_objects);
    }
};