#pragma once

#include <opencv2/opencv.hpp>
#include "my_utils.h"
#include "RealCam.h"

class YellowDetector
{
private:
    static const size_t W = RealCam::W;
    static const size_t H = RealCam::H;

    static int dx(size_t i) { static int val[4] = {-1,  0, 0, 1}; return val[i]; }
    static int dy(size_t i) { static int val[4] = { 0, -1, 1, 0}; return val[i]; }

    static const size_t BILATERAL_SIZE = 19;  // in pixel
    static const unsigned char H_DOWN  = 20;
    static const unsigned char C_DOWN  = 100;
    static const unsigned char V_DOWN  = 100;
    static const unsigned char H_UP    = 30;
    static const unsigned char C_UP    = 255;
    static const unsigned char V_UP    = 255;
    static const size_t FAN_SIZE       = 2;   // in pixel
    static const size_t FILTER_SIZE    = 200; // in pixel

    typedef std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector4f>> vectorV3;

    static Field<bool> find_yellow(const cv::Mat& img)
    {
        static cv::Mat blur;
        static cv::Mat hcv;
        static cv::Mat binar;

        cv::bilateralFilter(img, blur, BILATERAL_SIZE, 75, 75);
        cv::cvtColor(blur, hcv, cv::COLOR_BGR2HSV);
        cv::inRange(hcv, cv::Scalar(H_DOWN, C_DOWN, V_DOWN), cv::Scalar(H_UP, C_UP, V_UP), binar);

        static Field<bool> mask(W, H);
        mask.fill(false);
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        if (binar.at<unsigned char>(cv::Point(x, y)) == 255)
            mask(x, y) = true;

        return mask;
    }

    static void companents_fan(Field<bool>& mask, size_t d, const bool fan_type)
    {
        static std::vector<cv::Point> Q;
        static Field<int> dist(W, H);
        Q.clear();
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

    static Field<int> find_all_companents(const Field<bool>& mask, const bool val)
    {
        static Field<int> cmp(W, H);
        cmp.fill(-1);

        size_t num = 0;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (mask(x, y) != val) continue;
            if (cmp(x, y) != -1) continue;

            static std::vector<cv::Point> Q;
            Q = {cv::Point(x, y)};
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

    static vectorV3 make_vectors(const Field<bool>& mask, const RealCam& cam)
    {
        static Field<int> companents;
        static vectorV3 vectors;

        companents = find_all_companents(mask, true);

        std::vector<cv::Point> center;
        std::vector<size_t> cnt;
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            int num = companents(x, y);
            if (num == -1) continue;

            if (center.size() <= static_cast<size_t>(num)) 
            {
                center.resize(num + 1);
                cnt.resize(num + 1);
            }
            center[num].x += x;
            center[num].y += y;
            cnt[num]++;
        }

        size_t n = center.size();
        vectors.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            cv::Point p = cv::Point(center[i].x / cnt[i], center[i].y / cnt[i]);
            vectors[i] = cam.deproject(p);
        }

        return vectors;
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
    YellowDetector() = delete;

    static vectorV3 detect(const RealCam& cam)
    {
        cv::Mat img(cv::Size(W, H), CV_8UC3);
        static Field<bool> yellow(W, H);
        static vectorV3 vectors;

        cam.getImageColor(img);
        yellow = find_yellow(img);
        companents_fan(yellow, FAN_SIZE, true);
        companents_filter(yellow, true);
        vectors = make_vectors(yellow, cam);

        //======= draw
        static cv::Mat img_yellow(cv::Size(W, H), CV_8UC3);

        img_yellow.setTo(cv::Scalar(0,0,0));
        draw_mask(img_yellow, yellow, 1, 255);

        SingleImage::add(img);
        SingleImage::add(img_yellow);
        //======= draw

        return vectors;
    }
};