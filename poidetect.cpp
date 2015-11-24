#include "precomp.hpp"
/* 包含了主要的类 */
#include "cascadedetect.hpp" 
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
using namespace std;
using namespace cv;

std::string& trim(std::string &s) 
{
	if (s.empty()) 
	{
		return s;
	}

	s.erase(0,s.find_first_not_of(" \t"));
	s.erase(s.find_last_not_of(" \t") + 1);
	return s;
}

vector<string> strsplit(string &str, const string &delim)
{
	vector<string> result;
	size_t last = 0;
	size_t index = str.find_first_of(delim, last);
	while(index != string::npos)
	{
		if(index - last >0)
		{
			result.push_back(str.substr(last, index-last));
		}
		last = index+1;
		index = str.find_first_of(delim, last);
	}
	if(index - last>0)
	{
		result.push_back(str.substr(last, index-last));
	}
	return result;
}

bool startswith(string str, string substr)
{
	return str.find(substr)==0? true : false;
}

bool endswith(string str, string substr)
{
	int sub = str.length() - substr.length();
	if(sub <0)
		return false;
	return str.rfind(substr)==sub? true : false;
}

POIDetecter::POIDetecter(string configFilePath)
{
	POIDetecter(configFilePath, "");
}

POIDetecter::POIDetecter(string configFilePath, string mapfile)
{
	ifstream infile(configFilePath.c_str());
	if(!infile)
	{
		cout<< "config file is error" << endl;
	}
	size_t lastpos = configFilePath.find_last_of("\\/");
	string modelpath = configFilePath.substr(0, lastpos) + "/";

	string configline;
	while (getline(infile, configline))
	{
		configline = trim(configline);
		if(configline.empty())
		{
			continue;
		}
		vector<string> configitems = strsplit(configline, " \t");
		if(configitems.size() < 2)
		{
			continue;
		}
		int label = atoi(configitems[0].c_str());
		CascadeClassifier classifier;
		if(!classifier.load(modelpath + configitems[1]))
		{
			cout << "load model file fail, please check!" << endl;
		}
		classifierMap[label].push_back(classifier);
	}
	 
	if(!mapfile.empty())
	{
		loadWordMap(mapfile);
	}
}

bool POIDetecter::loadWordMap(string filename)
{
	ifstream conffile( filename.c_str() );
	if(!conffile)
	{
		cout<< "load word map fail!"<< endl;
		return false;
	}
	string configline;
	wordidMap.clear();
	idwordMap.clear();

	while(getline(conffile, configline))
	{
		configline = trim(configline);
		if(configline.empty())
		{
			continue;
		}
		vector<string> configitems = strsplit(configline, " \t");
		if(configitems.size() < 2)
		{
			continue;
		}

		string keystr = configitems[0];
		int value = atoi(configitems[1].c_str());
		wordidMap[keystr] = value;
		idwordMap[value] = keystr;
	}
	return true;
}

bool POIDetecter::getPoiNameMap(string poiname, vector<WordID> &poivec)
{
	poivec.clear();
	const char* pstr = poiname.c_str();
	char buf[4];
	memset( buf, 0, sizeof( buf));
	bool skip_start = false;
	for( ; *pstr != '\0'; ++ pstr)
	{
		if(*pstr == '(' )
		{
			skip_start = true;
			continue;
		}else if( *pstr == ')' )
		{
			skip_start = false;
			continue;
		}
		if( skip_start )
		{
			continue;
		}

		if( (*pstr & 0x80) == 0x80 )/* 在ANSI C标准中一个汉字由两个字节组成，判断一个字符是否为汉字就是判断第一个字节的最高位是否为1 */
		{
			buf[0] = *pstr;
			++ pstr;
			buf[1] = *pstr;
#ifdef __unix__
			++ pstr;
			buf[2] = *pstr;
#endif
		}else{
			buf[0] = *pstr;
			buf[1] = '\0';
		}
		int matchid = 0 ;
		if( getID(buf, matchid))
		{
			struct WordID wi = {matchid, buf};
			poivec.push_back(wi);
		}
	}
	return true;
}

vector<OutBlock> POIDetecter::detectPoi(string poipath, string poiname)
{
	vector<OutBlock> result;
	Mat image = imread(poipath.c_str());
	if(image.empty())
	{
		cout<< "image is null!" <<endl;
		return result;
	}

	poivec.clear();
	getPoiNameMap(poiname, poivec);
	/* 要使用的检测器 */
	classifierUsed.clear();
	for(size_t i=0; i< poivec.size(); i++)
	{
		int label = poivec[i].id;
		if(classifierMap.find(label) != classifierMap.end())
		{
			/* 有需要使用的分类器 */
			if(classifierUsed.find(label) != classifierUsed.end())
			{
				continue;
			}
			classifierUsed[label] = classifierMap[label];
		}
	}
	if(classifierUsed.empty())
	{
		return result;
	}
	/* 检测函数 */
	/* Mat image
	   vector<OutBlock>& objects,
	   double scaleFactor
	   int minNeighbors,
		Size minObjectSize,
		Size maxObjectSize */
	detectMultiScale( image, result, 1.15, 8, Size(20,20) );
	return result;
}

/* object_x是已经融合好的两个 */
map<int, vector<OutBlock> > POIDetecter::mergeResult(vector<OutBlock> objects_1, vector<OutBlock> objects_2)
{
	for(size_t j = 0; j< objects_2.size(); j++)
	{
		int label_2 = objects_2[j].label;
		bool hased = false;
		for(size_t i=0; i<objects_1.size(); i++)
		{
			int label_1 = objects_1[i].label;
			if(label_2 == label_1)
			{
				//juage
				if(judgeOver(objects_2[j], objects_1[i]))
				{
					objects_1[i].rect.x = (objects_2[j].rect.x + objects_1[i].rect.x)/2;
					objects_1[i].rect.y = (objects_2[j].rect.y + objects_1[i].rect.y)/2;
					objects_1[i].rect.width = (objects_2[j].rect.width + objects_1[i].rect.width)/2;
					objects_1[i].rect.height = (objects_2[j].rect.height + objects_1[i].rect.height)/2;
					//center
					objects_1[i].x_0 = (objects_2[j].x_0 + objects_1[i].x_0)/2;
					objects_1[i].y_0 = (objects_2[j].y_0 + objects_1[i].y_0)/2;
					hased = true;
				}
			}
		}
		if( !hased )
		{
			objects_1.push_back(objects_2[j]);
		}
	}

	map<int, vector<OutBlock> > label_objects;
	for(size_t i=0; i<objects_1.size(); i++)
	{
		int label= objects_1[i].label;
		objects_1[i].visit = false; // set false
		label_objects[label].push_back( objects_1[i] );
	}
	return label_objects;
}

bool POIDetecter::judgeOver(OutBlock &objects_1, OutBlock &objects_2)
{
	//1
	int xx = objects_1.rect.x;
	int yy = objects_1.rect.y;
	int ww = objects_1.rect.width;
	int hh = objects_1.rect.height;
	//2
	int xx2 = objects_2.rect.x;
	int yy2 = objects_2.rect.y;
	int ww2 = objects_2.rect.width;
	int hh2 = objects_2.rect.height;
	//judge
	if(xx + ww < xx2 || xx > xx2 + ww2)
		return false;
	if(yy + hh < yy2 || yy > yy2 + hh2)
		return false;
	int x_left = xx2 < xx? xx : xx2;
	int x_right = ( xx2 + ww2) > ( xx + ww ) ? ( xx + ww ) : (xx2 + ww2);
	int y_top = yy2 < yy ? yy : yy2;
	int y_bottom = ( yy2+hh2 ) > ( yy+hh ) ? ( yy+hh ) : (yy2 + hh2);
	int cover_area = ( x_right - x_left ) * ( y_bottom - y_top );
	if (cover_area*4 > ww*hh || cover_area*4 > ww2*hh2)
	{
		return true;
	}
	return false;
}

//object is real
bool POIDetecter::judgeCondition(vector<OutBlock> &path, OutBlock &object)
{
	if(path.empty())
		return true;
	if(object.visit)
		return false;
	int obj_x = object.rect.x;
	int obj_y = object.rect.y;
	int obj_w = object.rect.width;
	int obj_h = object.rect.height;
	int nodesize = path.size();
	OutBlock before_ele(-1);
	for(int i = nodesize -1; i>=0; i--)
	{
		if(path[i].label >=0)
		{
			before_ele = path[i];
			break;
		}
	}
	//0: visit
	if(before_ele.label >=0 && ( before_ele.x_0 > object.x_0 && before_ele.y_0 > object.y_0 )) 
		return false;
	//1: area size
	int total_area = obj_w * obj_h;

	int realsize = 1;
	for(size_t i = 0; i< nodesize; i++)
	{
		if(path[i].label < 0)
			continue;
		realsize++;
		total_area += path[i].rect.width * path[i].rect.height;
	}
	if(realsize == 1)
	{
		// ele of path all is -1
		return true;
	}
	float threshold = (float)total_area / realsize; 
	if(obj_w * obj_h >4*threshold || obj_w * obj_h * 4 < threshold)
		return false;
	for(size_t i =0; i < nodesize; i++)
	{
		if(path[i].label < 0)
			continue;
		int nowArea = path[i].rect.width * path[i].rect.height;
		if( nowArea > 4*threshold || nowArea*4 < threshold )
		{
			return false;
		}
	}
	//2:area over
	for(size_t i =0; i< nodesize; i++)
	{
		if(path[i].label < 0)
			continue;
		if(judgeOver(path[i], object))
			return false;
	}
	//3.some area
	int min_x = INT_MAX,min_y = INT_MAX,max_w = -1,max_h = -1;
	for(size_t i=0; i<nodesize; i++)
	{
		if(path[i].label < 0)
			continue;
		Rect rect = path[i].rect;
		if(rect.x < min_x)
			min_x = rect.x;
		if(rect.y < min_y)
			min_y = rect.y;
		if(rect.x + rect.width > max_w)
			max_w = rect.x + rect.width;
		if(rect.y + rect.height > max_h)
			max_h = rect.y + rect.height;
	}
	if( obj_x + obj_w < min_x && obj_y + obj_h < min_y )
		return false;
	if( obj_x > max_w && obj_y + obj_h < min_y )
		return false;
	if( obj_x + obj_w < min_x && obj_y > max_h)
		return false;
	if( obj_x > max_w && obj_y > max_h)
		return false;
	return true;
}

bool POIDetecter::pathEmpty(vector<OutBlock> &path)
{
	if(path.empty())
		return true;
	bool allEmptyNode = true;
	for(size_t i = 0; i< path.size(); i++)
	{
		if(path[i].label >=0)
		{
			allEmptyNode = false;
			break;
		}
	}
	return allEmptyNode;
}

vector<OutBlock> POIDetecter::judgeResult( map<int, vector<OutBlock> > &objects)
{
	//posResult.clear();
    vector<vector<OutBlock> > v;
    posResult.swap(v);
	vector<OutBlock> pathResult;
	if(objects.empty())
		return pathResult;

    /*for(map<int, vector<OutBlock> >::iterator it = objects.begin(); it!=objects.end(); it++)
    {
        vector<OutBlock> tt = it->second;
        cout<< "case:" << it->first << " " << tt.size() <<endl;
        for(int kk =0; kk< tt.size(); kk++)
        {
            cout << tt[kk].rect.x << " " << tt[kk].rect.y << " " <<tt[kk].rect.width << " " << tt[kk].rect.height << " " << tt[kk].label << " " << tt[kk].visit << endl;
        }
    }*/
	OutBlock anyBlock(-1);
	for(size_t i =0; i < poivec.size(); i++)
	{
		int targetId = poivec[i].id;
		objects[targetId].push_back(anyBlock);
	}
	recurionResult(objects, 0, vector<OutBlock>());
	posResult.pop_back();
	// min area
	
	unsigned int maxlen = 0;
	for(size_t i =0; i< posResult.size(); i++)
	{
		vector<OutBlock> pospath;
		for(size_t j =0; j< posResult[i].size(); j++)
		{
			if(posResult[i][j].label >=0)
			{
				pospath.push_back(posResult[i][j]);
			}
		}
		if(pospath.size() > maxlen)
		{
			pathResult = pospath;
			maxlen = pospath.size();
		}
	}

	// build string
	return pathResult;
}

/* objects:所有的框框 */
void POIDetecter::recurionResult(map<int, vector<OutBlock> > objects, int position, vector<OutBlock> pospath)
{
	if(objects.empty() || posResult.size() > 50000) // enough cases
	{
		return;
	}

	if(position == poivec.size())
	{
		posResult.push_back(pospath);
		return;
	}

	int target_label = poivec[position++].id;

	for(size_t i=0; i<objects[target_label].size(); i++)
	{
		OutBlock target =  objects[target_label][i];
		if(target.label < 0)
		{
			pospath.push_back(objects[target_label][i]);
			recurionResult(objects, position, pospath);
			pospath.pop_back();
		}
		else if(judgeCondition(pospath, target))
		{
			objects[target_label][i].visit = true;

			pospath.push_back(objects[target_label][i]);
			recurionResult(objects, position, pospath);
			pospath.pop_back();

			objects[target_label][i].visit = false;
		}
	}


}

int POIDetecter::calculateArea(vector<OutBlock> blocks)
{
	if(blocks.empty())
	{
		return 0;
	}
	int min_x = INT_MAX,min_y = INT_MAX,max_w = -1,max_h = -1;
	for(size_t i=0; i<blocks.size(); i++)
	{
		Rect rect = blocks[i].rect;
		if(rect.x < min_x)
			min_x = rect.x;
		if(rect.y < min_y)
			min_y = rect.y;
		if(rect.x + rect.width > max_w)
			max_w = rect.x + rect.width;
		if(rect.y + rect.height > max_h)
			max_h = rect.y + rect.height;
	}
	int block_area = (max_w - min_x)*(max_h - min_y);
	return block_area;
}

bool POIDetecter::empty()
{
	bool result = false;
	/* 所有的分类器都不能为空 */
	std::map<int ,vector<CascadeClassifier> >::iterator it;
	for(it = classifierUsed.begin(); it != classifierUsed.end(); it++)
	{
		vector<CascadeClassifier> classifiers = it->second;
		for(size_t i =0; i < classifiers.size(); i++)
		{
			if(classifiers[i].data.stages.empty())
			{
				result = true;
				break;
			}
		}
		if(result)
			break;
	}
	return result;
}

double tt = 0.0;
/* 检测函数 */
void POIDetecter::detectMultiScale( const Mat& image, vector<OutBlock>& objects,
                                          vector<int>& rejectLevels,
                                          vector<double>& levelWeights,
                                          double scaleFactor, int minNeighbors,
                                          Size minObjectSize, Size maxObjectSize,
                                          bool outputRejectLevels )
{
    const double GROUP_EPS = 0.2;
    CV_Assert( scaleFactor > 1 && image.depth() == CV_8U );

    if( empty() )
        return;
    objects.clear();

    if( maxObjectSize.height == 0 || maxObjectSize.width == 0 )
	{
		maxObjectSize.width = image.size().width/2;
		maxObjectSize.height = image.size().height/2;
	}

    Mat grayImage = image;
    if( grayImage.channels() > 1 )
    {
        Mat temp;
        cvtColor(grayImage, temp, CV_BGR2GRAY);
        grayImage = temp;
    }
    
	std::map<int ,vector<CascadeClassifier> > scaleClassifiers;
	/* 不同分类器多个尺度 */
	vector<Rect> candidates;
	map<int , vector<Rect> > hz_rect;
	
	double factor = 1;
	if(image.rows * image.cols < 1e6)
	{
		/* 100万像素一下的扩大一下 */
		/* factor = 24/32*/
		factor = 0.75;
	}
	Mat imageBuffer(image.rows/factor + 2, image.cols/factor + 2, CV_8U);
	/* 计算步长 */
	int yStep = 4;

	double theFactor = scaleFactor;
	for( ; ; factor *= theFactor )
	{
		//if(factor < 1.0)
		//	theFactor = 1.17; // 0.625*1.17*1.17*1.17=1
		//else
		//	theFactor = scaleFactor;

		scaleClassifiers.clear();
		Size scaledImageSize( cvRound( grayImage.cols/factor ), cvRound( grayImage.rows/factor ) );
		/* 处理各个字符的分类器 */
		for(std::map<int ,vector<CascadeClassifier> >::iterator it = classifierUsed.begin(); it != classifierUsed.end(); it++)
		{
			int keylabel = it->first;
			vector<CascadeClassifier> classifiers = it->second;
			for(size_t i =0; i < classifiers.size(); i++)
			{
				CascadeClassifier classifier = classifiers[i];

				Size originalWindowSize = classifier.getOriginalWindowSize();
				Size windowSize( cvRound(originalWindowSize.width*factor), cvRound(originalWindowSize.height*factor) );
				Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );

				if( processingRectSize.width <= 0 || processingRectSize.height <= 0 )/* 扩大后图像的尺寸小于分类器窗尺寸 */
					continue;
				if( windowSize.width > maxObjectSize.width || windowSize.height > maxObjectSize.height )
					continue;
				if( windowSize.width < minObjectSize.width && windowSize.height < minObjectSize.height )/* 最小尺寸的设置在多宽高比下要谨慎 */
					continue;
				scaleClassifiers[keylabel].push_back(classifier);
			}
		}
		/* 改尺寸下不存在任何分类器 */
		if(scaleClassifiers.empty())
		{
			break;
		}

		/* 缩放图片 */
		Mat scaledImage( scaledImageSize, CV_8U, imageBuffer.data );
		resize( grayImage, scaledImage, scaledImageSize, 0, 0, CV_INTER_LINEAR );
		int stripCount, stripSize;

		/* 得到haar所需要的积分图 */
		int cn = scaledImageSize.width +1, rn = scaledImageSize.height+1;
		Mat sum_scale = Mat(rn, cn, CV_32S);
		Mat sqsum_scale = Mat(rn, cn, CV_64F);
		Mat	tilted_scale = Mat(rn, cn, CV_32S);
		//integral(scaledImage, sum_scale, sqsum_scale, tilted_scale);

		//hog
		vector<Mat> hist_scale;
		hist_scale.clear();
		for( int bin = 0; bin < HOGEvaluator::Feature::BIN_NUM; bin++ )
		{
			hist_scale.push_back( Mat(rn, cn, CV_32FC1) );
		}
		Mat normSum_scale;
		normSum_scale.create( rn, cn, CV_32FC1 );
		integralHistogram( scaledImage, hist_scale, normSum_scale, HOGEvaluator::Feature::BIN_NUM );

		const int PTS_PER_THREAD = 1000;
		//double t0 = (double)getTickCount();
		//tt = 0.0;
		for(std::map<int ,vector<CascadeClassifier> >::iterator it = scaleClassifiers.begin(); it != scaleClassifiers.end(); it++)
		{
			int keylabel = it->first;
			vector<CascadeClassifier> classifiers = it->second;
			for(size_t i =0; i < classifiers.size(); i++)
			{
				CascadeClassifier classifier = classifiers[i];
				Size originalWindowSize = classifier.getOriginalWindowSize();
				Size processingRectSize( scaledImageSize.width - originalWindowSize.width, scaledImageSize.height - originalWindowSize.height );
				/* 并行个数以及大小，按照列进行并行处理，确实是列 */
				stripCount = ((processingRectSize.width/yStep)*(processingRectSize.height + yStep-1)/yStep + PTS_PER_THREAD/2)/PTS_PER_THREAD;
				stripCount = std::min(std::max(stripCount, 1), 100);
				stripSize = (((processingRectSize.height + stripCount - 1)/stripCount + yStep-1)/yStep)*yStep;
				/* 调用单尺度检测函数进行检测 */
				/* yStep是步长，factor是因子 */
				candidates.clear();
				/* candidates是结果 */
				if( !classifier.detectSingleScale( &sum_scale, &sqsum_scale, &tilted_scale,
					hist_scale, normSum_scale,
					scaledImage, stripCount, processingRectSize, 
					stripSize, yStep, factor, candidates,
					rejectLevels, levelWeights, outputRejectLevels) )
					continue;
				/* 返回的结果 */
				for(size_t m=0; m< candidates.size(); m++)
				{		
					hz_rect[keylabel].push_back(candidates[m]);
				}
			}
		}
		//double exec_time = ((double)getTickCount() - t0)/getTickFrequency();
		//cout << "time: " << exec_time << endl;
	}
    //objects.resize(candidates.size());
    //std::copy(candidates.begin(), candidates.end(), objects.begin());
	/* 合并检测结果 */
	for(map<int, vector<Rect> >::iterator it = hz_rect.begin(); it != hz_rect.end(); it++)
	{
		int key = it->first;
		//vector<Rect> key_vec = it->second;
		groupRectangles( it->second, minNeighbors, GROUP_EPS );
	}

	for(map<int, vector<Rect> >::iterator it = hz_rect.begin(); it != hz_rect.end(); it++)
	{
		int key = it->first;
		string hz;
		getWord(key, hz);
		vector<Rect> key_vec = it->second;
		for(size_t m = 0; m< key_vec.size(); m++)
		{
			objects.push_back(OutBlock(key_vec[m], key, hz));
		}
	}
}

void POIDetecter::detectMultiScale( const Mat& image, vector<OutBlock>& objects,
                                          double scaleFactor, int minNeighbors,
                                          Size minObjectSize, Size maxObjectSize)
{
    vector<int> fakeLevels;
    vector<double> fakeWeights;
    detectMultiScale( image, objects, fakeLevels, fakeWeights, scaleFactor,
        minNeighbors, minObjectSize, maxObjectSize, false );
}

void POIDetecter::integralHistogram(const Mat &img, vector<Mat> &histogram, Mat &norm, int nbins)
{
	CV_Assert( img.type() == CV_8U || img.type() == CV_8UC3 );
	int x, y, binIdx;

	Size gradSize(img.size());
	Size histSize(histogram[0].size());
	Mat grad(gradSize, CV_32F);
	Mat qangle(gradSize, CV_8U);

	AutoBuffer<int> mapbuf(gradSize.width + gradSize.height + 4);
	int* xmap = (int*)mapbuf + 1;
	int* ymap = xmap + gradSize.width + 2;

	const int borderType = (int)BORDER_REPLICATE;

	for( x = -1; x < gradSize.width + 1; x++ )
		xmap[x] = borderInterpolate(x, gradSize.width, borderType);
	for( y = -1; y < gradSize.height + 1; y++ )
		ymap[y] = borderInterpolate(y, gradSize.height, borderType);

	int width = gradSize.width;
	AutoBuffer<float> _dbuf(width*4);
	float* dbuf = _dbuf;
	Mat Dx(1, width, CV_32F, dbuf);
	Mat Dy(1, width, CV_32F, dbuf + width);
	Mat Mag(1, width, CV_32F, dbuf + width*2);
	Mat Angle(1, width, CV_32F, dbuf + width*3);

	float angleScale = (float)(nbins/CV_PI);

	for( y = 0; y < gradSize.height; y++ )
	{
		const uchar* currPtr = img.data + img.step*ymap[y];
		const uchar* prevPtr = img.data + img.step*ymap[y-1];
		const uchar* nextPtr = img.data + img.step*ymap[y+1];
		float* gradPtr = (float*)grad.ptr(y);
		uchar* qanglePtr = (uchar*)qangle.ptr(y);

		for( x = 0; x < width; x++ )
		{
			dbuf[x] = (float)(currPtr[xmap[x+1]] - currPtr[xmap[x-1]]);
			dbuf[width + x] = (float)(nextPtr[xmap[x]] - prevPtr[xmap[x]]);
		}
		cartToPolar( Dx, Dy, Mag, Angle, false );
		for( x = 0; x < width; x++ )
		{
			float mag = dbuf[x+width*2];
			float angle = dbuf[x+width*3];
			angle = angle*angleScale - 0.5f;
			int bidx = cvFloor(angle);
			angle -= bidx;
			if( bidx < 0 )
				bidx += nbins;
			else if( bidx >= nbins )
				bidx -= nbins;

			qanglePtr[x] = (uchar)bidx;
			gradPtr[x] = mag;
		}
	}
	/* norm的值 */
	integral(grad, norm, grad.depth());

	float* histBuf;
	const float* magBuf;
	const uchar* binsBuf;

	int binsStep = (int)( qangle.step / sizeof(uchar) );
	int histStep = (int)( histogram[0].step / sizeof(float) );
	int magStep = (int)( grad.step / sizeof(float) );
	for( binIdx = 0; binIdx < nbins; binIdx++ )
	{
		histBuf = (float*)histogram[binIdx].data;
		magBuf = (const float*)grad.data;
		binsBuf = (const uchar*)qangle.data;

		memset( histBuf, 0, histSize.width * sizeof(histBuf[0]) );
		histBuf += histStep + 1;
		for( y = 0; y < qangle.rows; y++ )
		{
			histBuf[-1] = 0.f;
			float strSum = 0.f;
			for( x = 0; x < qangle.cols; x++ )
			{
				if( binsBuf[x] == binIdx )
					strSum += magBuf[x];
				histBuf[x] = histBuf[-histStep + x] + strSum;
			}
			histBuf += histStep;
			binsBuf += binsStep;
			magBuf += magStep;
		}
	}
}

void POIDetecter::groupRectangles(vector<Rect>& rectList, int groupThreshold, double eps)
{
	if( groupThreshold <= 0 || rectList.empty() )
	{
		return;
	}
	vector<int> labels;
	/* 对rectList中的矩形进行分类 */
	int nclasses = partition(rectList, labels, SimilarRects(eps));

	vector<Rect> rrects(nclasses);
	vector<int> rweights(nclasses, 0);
	int i, j, nlabels = (int)labels.size();
	/* 组合分到同一类别的矩形并保存当前类别下通过stage的最大值以及最大的权重 */
	for( i = 0; i < nlabels; i++ )
	{
		int cls = labels[i];
		rrects[cls].x += rectList[i].x;
		rrects[cls].y += rectList[i].y;
		rrects[cls].width += rectList[i].width;
		rrects[cls].height += rectList[i].height;
		rweights[cls]++;
	}

	for( i = 0; i < nclasses; i++ )
	{
		Rect r = rrects[i];
		float s = 1.f/rweights[i];
		rrects[i] = Rect(saturate_cast<int>(r.x*s),
			saturate_cast<int>(r.y*s),
			saturate_cast<int>(r.width*s),
			saturate_cast<int>(r.height*s));
	}

	vector<float> normweights(nclasses, 0);
	/* 阈值的处理，阈值表示最多保留几个 */
	for( i=0; i < nclasses; i++)
	{
		float areas = (float)rrects[i].width * rrects[i].height;
		float normweight = (float)rweights[i] / areas;
		normweights[i] = normweight;
	}
	rectList.clear();
	/* 按照groupThreshold合并规则，以及是否存在包含关系输出合并后的矩形 */
	for( i = 0; i < nclasses; i++ )
	{
		Rect r1 = rrects[i];
		int n1 = rweights[i];

		if( n1 <= groupThreshold )
			continue;
		// filter out small face rectangles inside large rectangles
		for( j = 0; j < nclasses; j++ )
		{
			int n2 = rweights[j];
			if( j == i || n2 <= groupThreshold )
				continue;
			Rect r2 = rrects[j];

			int dx = saturate_cast<int>( r2.width * eps );
			int dy = saturate_cast<int>( r2.height * eps );
			/* 当r1在r2的内部的时候，停止 */
			if( i != j &&
				r1.x >= r2.x - dx &&
				r1.y >= r2.y - dy &&
				r1.x + r1.width <= r2.x + r2.width + dx &&
				r1.y + r1.height <= r2.y + r2.height + dy &&
				(n2 > std::max(3, n1) || n1 < 3) )
				break;
		}

		if( j == nclasses )
		{
			rectList.push_back(r1);
		}
	}
}
