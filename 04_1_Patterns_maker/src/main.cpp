#include <iostream>
#include <ctime> 
#include <vector>
#include <fstream>
#include "RealCam.h"

using namespace std;
using namespace Eigen;

const size_t W = RealCam::W;
const size_t H = RealCam::H;
const size_t INIT_TRESH_BOARD = 40;
const string CONFIG = "data/config.txt";
const string NAME = "data/M20_100_VU.ptr";

const size_t X = 80;
const size_t Y = 80;
size_t D = 1100;

void save(cv::Mat& binar)
{
    cv::cvtColor(binar, binar, CV_BGR2GRAY);

    vector<cv::Point> points;
    for (size_t x = W/2-X; x <= W/2+X; x++)
    for (size_t y = H/2-Y; y <= H/2+Y; y++)
        if (binar.at<unsigned char>(cv::Point(x, y)) == 0)
            points.push_back(cv::Point(x, y));

    ofstream fout(NAME, std::ios_base::binary);

    size_t sz = points.size();
    fout.write(reinterpret_cast<char*>(&sz), sizeof(size_t));
    for (cv::Point p : points)
    {
        fout.write(reinterpret_cast<char*>(&p.x), sizeof(int));
        fout.write(reinterpret_cast<char*>(&p.y), sizeof(int));
    }

    fout.close();

    ofstream fcfg(CONFIG, std::ios_base::ate);
    fcfg << NAME << endl;
    fcfg.close();
}

void save(RealCam& cam)
{
    vector<cv::Point> points;
    for (size_t x = W/2-X; x <= W/2+X; x++)
    for (size_t y = H/2-Y; y <= H/2+Y; y++)
    {
        if (cam.depth(x, y).d < 10) continue;
        if (cam.depth(x, y).d > D) continue;
        
        points.push_back(cv::Point(x, y));
    }

    ofstream fout(NAME, std::ios_base::binary);

    size_t sz = points.size();
    fout.write(reinterpret_cast<char*>(&sz), sizeof(size_t));
    for (cv::Point p : points)
    {
        fout.write(reinterpret_cast<char*>(&p.x), sizeof(int));
        fout.write(reinterpret_cast<char*>(&p.y), sizeof(int));
    }

    fout.close();

    ofstream fcfg(CONFIG, std::ios_base::ate);
    fcfg << NAME << endl;
    fcfg.close();
}

int main()
{
    srand(time(NULL));

    RealCam cam;
    cv::Mat img(cv::Size(W, H), CV_8UC3);
    cv::Mat depth(cv::Size(W, H), CV_8UC3);
    cv::Mat gray(cv::Size(W, H), CV_8UC3);
    cv::Mat binar;
    cv::Mat usable(cv::Size(W, H), CV_8UC3);

    int tresh_board = INIT_TRESH_BOARD;
    SingleImage::init(W, H, 2, 2);
    while(true)
    {
        char c = cv::waitKey(1);
        if (c == 27) break;
        //if (c == '1') tresh_board--;
        //if (c == '2') tresh_board++;
        if (c == '1') D -= 10;
        if (c == '2') D += 10;
        if (c == 10) 
        {
            save(cam);
            cout << "save" << endl;
        }
        cout << D << endl;

        cam.update();
        cam.getImageColor(img);
        cam.getImageDepth(depth);

        usable.setTo(cv::Scalar(0,0,0));
        for (size_t x = 0; x < W; x++)
        for (size_t y = 0; y < H; y++)
        {
            if (cam.depth(x, y).d < 10) continue;
            if (cam.depth(x, y).d > D) continue;
            
            usable.at<cv::Vec3b>(cv::Point(x, y))[0] = 255;
        }

        //for (size_t x = 0; x < W; x++)
        //for (size_t y = 0; y < H; y++)
        //    gray.at<unsigned char>(cv::Point(x, y)) = img.at<cv::Vec3b>(cv::Point(x, y))[2];

        //cv::cvtColor(img, gray, CV_BGR2GRAY);
        //cv::threshold(gray, binar, tresh_board, 255, CV_THRESH_BINARY);

        //cv::cvtColor(gray, gray, CV_GRAY2BGR);
        //cv::cvtColor(binar, binar, CV_GRAY2BGR);

        cv::rectangle(depth, cv::Point(W/2-X, H/2-Y), cv::Point(W/2+X, H/2+Y), CV_RGB(0,0,0));

        SingleImage::add(img);
        //SingleImage::add(gray);
        //SingleImage::add(binar);
        SingleImage::add(depth);
        SingleImage::add(usable);
        SingleImage::show();
        SingleImage::clear();
    }

    return 0;
}
