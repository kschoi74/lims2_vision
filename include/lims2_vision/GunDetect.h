#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

Rect Get_Gun(const Mat & img, const Rect & HumanROI);
bool Get_Gun(const Mat & limg, const Mat & rimg,
             Rect lROI, Rect rROI, 
             Rect & lGunROI, Rect & rGunROI);
