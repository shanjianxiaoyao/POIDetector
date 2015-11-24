#ifndef __COMMON_H__
#define __COMMON_H__

#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;

struct OutBlock
{
	Rect rect;
	int label;
	string hanzi;
	bool visit;
	//center
	float x_0, y_0;
	OutBlock(Rect &r, int &b, string &s): rect(r), label(b), hanzi(s),visit(false)
	{
		x_0 = r.x + (float)r.width/2.0;
		y_0 = r.y + (float)r.height/2.0;
	}
	OutBlock(int b): label(b){}
};
#endif
