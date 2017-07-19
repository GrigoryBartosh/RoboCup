#include <iostream>
#include <ctime> 
#include "RealCam.h"
#include "ObjectRecognizer.h"

using namespace std;
using namespace Eigen;

const size_t W = RealCam::W;
const size_t H = RealCam::H;

int main()
{
    srand(time(NULL));

    RealCam cam;
    ObjectRecognizer recognizer;

    SingleImage::init(W, H, 2, 2);
    while(cv::waitKey(1) != 27)
    {
        cam.update();

        recognizer.get_object(cam);
        
        SingleImage::show();
        SingleImage::clear();
    }

    return 0;
}
