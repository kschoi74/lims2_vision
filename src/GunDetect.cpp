#include <lims2_vision/GunDetect.h>
#include <lims2_vision/lims2_vision_global.h>
#include <ros/ros.h>

using namespace ros;

// Get a bounding box of a gun within the given ROI
Rect Get_Gun(const Mat & img, const Rect & HumanROI)
{
	Mat HROI;
	Mat blob;
	Mat channels[3];
	Mat profile[2];

	int left = -1;
	int right;
	int top;
	int bottom;

	cvtColor(img(HumanROI), HROI, COLOR_RGB2Lab);
	split(HROI, channels);

	channels[0] = (channels[1] - channels[2]);

	threshold(channels[0], blob, 40, 255, cv::THRESH_BINARY);
	morphologyEx(blob, blob, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));

	reduce(blob, profile[0], 0, REDUCE_MAX, blob.type());	// a row
	reduce(blob, profile[1], 1, REDUCE_MAX, blob.type());	// a column

	int idx = 0;
	int cols = blob.cols;
	int rows = blob.rows;
	for (auto x = profile[0].begin<uchar>(); idx < cols ; x++, idx++) {
		if (*x == 255) break;
	}
	left = idx;

	if (left == -1)
		return Rect(-1, -1, -1, -1);

	idx = cols;
	for (auto x = profile[0].end<uchar>() - 1; idx > 0 ; x--, idx--) {
		if (*x == 255) break;
	}
	right = idx;

	idx = 0;
	for (auto x = profile[1].begin<uchar>(); idx < rows ; x++, idx++) {
		if (*x == 255) break;
	}
	top = idx;

	idx = rows;
	for (auto x = profile[1].end<uchar>() - 1; idx > 0 ; x--, idx--)	{
		if (*x == 255) break;
	}
	bottom = idx;

	int width = right - left;
	int height = bottom - top;

	if ( 0 < width && 0 < height )
		return Rect(HumanROI.x + left, HumanROI.y + top, width, height);
	else
		return Rect(-1, -1, -1, -1);
}

bool Get_Gun(const Mat & limg, const Mat & rimg, Rect lROI, Rect rROI, Rect & lGunROI, Rect & rGunROI)
{
	Mat HROI;
	Mat blob[2];
	Mat channels[3];
	Mat profile[2][2];

	int left[2];
	int right[2];
	int top = -1;
	int bottom;

	int ht = max(lROI.y, rROI.y);
	int hb = min(lROI.y + lROI.height, rROI.y + rROI.height);
	int hh = hb - ht;

	lROI.y = ht;
	lROI.height = hh;
	rROI.y = ht;
	rROI.height = hh;
	
	//	left
	cvtColor(limg(lROI), HROI, COLOR_RGB2Lab);
	split(HROI, channels);
	channels[0] = (channels[1] - channels[2]);

	threshold(channels[0], blob[0], 40, 255, cv::THRESH_BINARY);
	morphologyEx(blob[0], blob[0], MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));
	// 	right
	cvtColor(rimg(rROI), HROI, COLOR_RGB2Lab);
	split(HROI, channels);
	channels[0] = (channels[1] - channels[2]);

	threshold(channels[0], blob[1], 40, 255, cv::THRESH_BINARY);
	morphologyEx(blob[1], blob[1], MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)));

	// for vertical search	
	reduce(blob[0], profile[0][1], 1, REDUCE_MAX, blob[0].type());	// a column	
	reduce(blob[1], profile[1][1], 1, REDUCE_MAX, blob[1].type());	// a column

	// vertical search
	int idx;
	int rows = blob[0].rows;
	ROS_ASSERT( rows == blob[1].rows );
	auto x0 = profile[0][1].begin<uchar>();
	auto x1 = profile[1][1].begin<uchar>();
	for ( idx = 0 ; idx < rows ; x0++, x1++, idx++ ) {
		if (*x0 == 255 && *x1 == 255) { // when both are detected
			top = idx;
			break;	
		}
	}

	if (top == -1) {
		lGunROI = 
		rGunROI = Rect(-1, -1, -1, -1);
	 	return false;
	}

	x0 = profile[0][1].end<uchar>() - 1;
	x1 = profile[1][1].end<uchar>() - 1;
	for ( idx = rows - 1 ; idx >= 0 ; x0--, x1--, idx-- ) {
		if (*x0 == 255 && *x1 == 255) {
			bottom = idx;
			break;
		}
	}

	ROS_ASSERT( blob[0].rows >= bottom );
	ROS_ASSERT( top <= bottom );
	reduce(blob[0].rowRange( top, bottom ), profile[0][0], 0, REDUCE_MAX, blob[0].type());	// a row
	reduce(blob[1].rowRange( top, bottom ), profile[1][0], 0, REDUCE_MAX, blob[1].type());	// a row

	// horizontal search
	x0 = profile[0][0].begin<uchar>();	
	for ( idx = 0 ; idx < blob[0].cols ; x0++, idx++ ) {
		if (*x0 == 255) {
			left[0] = idx;
			break;
		}
	}
	x0 = profile[0][0].end<uchar>() - 1;
	for (idx = blob[0].cols - 1 ; idx >= 0 ; x0--, idx--) {
		if (*x0 == 255) {
			right[0] = idx;
			break;
		}
	}
	//
	x1 = profile[1][0].begin<uchar>();	
	for ( idx = 0 ; idx < blob[1].cols ; x1++, idx++ ) {
		if (*x1 == 255) {
			left[1] = idx;
			break;
		}
	}
	x1 = profile[1][0].end<uchar>() - 1;
	for (idx = blob[1].cols - 1 ; idx >= 0 ; x1--, idx--) {
		if (*x1 == 255) {
			right[1] = idx;
			break;
		}
	}

	int width[2] = { right[0] - left[0], right[1] - left[1] };	
	int height = bottom - top;
	if ( 0 < width[0] && 0 < width[1] && 0 < height )
	{
		lGunROI = Rect(lROI.x + left[0], ht + top, width[0], height);
		rGunROI = Rect(rROI.x + left[1], ht + top, width[1], height);
		return true;
	}
	
	lGunROI = rGunROI = Rect(-1, -1, -1, -1);
	return false;
}
