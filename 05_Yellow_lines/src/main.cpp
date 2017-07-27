#include <iostream>
#include <ctime>
#include "RealCam.h"
#include "YellowDetector.h"

using namespace std;
using namespace Eigen;

const size_t W = RealCam::W;
const size_t H = RealCam::H;

const size_t BILATERAL_SIZE = 19;

int main()
{
    srand(time(NULL));

    RealCam cam;

    SingleImage::init(W, H, 2, 2);
    while(cv::waitKey(1) != 27)
    {
        cam.update();

        YellowDetector::detect(cam);

        SingleImage::show();
        SingleImage::clear();
    }

    return 0;
}
