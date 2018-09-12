#include <lims2_vision/GunDetect.h>

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
	for (auto x = profile[1].begin<uchar>(); idx < cols ; x++, idx++) {
		if (*x == 255) break;
	}
	top = idx;

	idx = rows;
	for (auto x = profile[1].end<uchar>() - 1; idx > 0 ; x--, idx--)	{
		if (*x == 255) break;
	}
	bottom = idx;

	return Rect(HumanROI.x + left, HumanROI.y + top, right - left, bottom - top);
}

