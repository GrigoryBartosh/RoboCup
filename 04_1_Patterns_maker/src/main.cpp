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
const string NAME = "data/S40_40_H.ptr";

const size_t X = 300;
const size_t Y = 220;

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

int main()
{
    srand(time(NULL));

    RealCam cam;
    cv::Mat img(cv::Size(W, H), CV_8UC3);
    cv::Mat gray(cv::Size(W, H), CV_8UC3);
    cv::Mat binar;

    int tresh_board = INIT_TRESH_BOARD;
    SingleImage::init(W, H, 2, 2);
    while(true)
    {
        char c = cv::waitKey(1);
        if (c == 27) break;
        if (c == '1') tresh_board--;
        if (c == '2') tresh_board++;
        if (c == 10) 
        {
            save(binar);
            cout << "save" << endl;
        }
        cout << tresh_board << endl;

        cam.update();
        cam.getImageColor(img);

        //for (size_t x = 0; x < W; x++)
        //for (size_t y = 0; y < H; y++)
        //    gray.at<unsigned char>(cv::Point(x, y)) = img.at<cv::Vec3b>(cv::Point(x, y))[2];

        cv::cvtColor(img, gray, CV_BGR2GRAY);
        cv::threshold(gray, binar, tresh_board, 255, CV_THRESH_BINARY);

        cv::cvtColor(gray, gray, CV_GRAY2BGR);
        cv::cvtColor(binar, binar, CV_GRAY2BGR);

        cv::rectangle(img, cv::Point(W/2-X, H/2-Y), cv::Point(W/2+X, H/2+Y), CV_RGB(0,0,0));
        putText(img, "lol kek\n cheburek", cv::Point(320, 240), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(255,255,255));

        SingleImage::add(img);
        SingleImage::add(gray);
        SingleImage::add(binar);     
        SingleImage::show();
        SingleImage::clear();
    }

    return 0;
}
