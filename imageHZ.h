#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/ml/ml.hpp>
#include "common.h"
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

class SVMDetector
{
public:
	vector<CvSVM *> svmModels;
	vector<string> wordMap;

	SVMDetector(string mapFileName, string modelsDir)
	{
		readMapFile(mapFileName); 
		initialModels(modelsDir);
	}

	~SVMDetector()
	{
		for (int i = 0; i < svmModels.size(); ++i)
		{
			if (svmModels[i] != NULL)
			{
				delete svmModels[i];
				svmModels[i] = NULL;
			}
		}
	}

	vector<OutBlock> svmDetectPOI(string poipath, string poiname);

private:
	void readMapFile(string mapFileName);
	/* 需要一次性加载的Model序号,序号从小到大排列 */
	vector<int> readInitialModelSerials(string fileName);

	void initialModels(string modelsDir,int partLoadFlag = 0 );
	void getUseModel(string imgName, vector<string>& useSVMModelNames, vector<int>& useSVMSerial,
		vector<string> &wordMap, string modelDir);

};

class boxHZ{
public:
	vector<Point> vMaxP,vMinP;
	vector<double> xyRatio;
	vector<int> xlen;
	vector<int> ylen;
	vector<Vec3d> Color;	
	vector<int> Region;
	vector<int> Size;
	vector<Point> Position;
	vector<int> edge;
	vector<double> edgeRatio;/* 为1s添加 */
	vector<int> sharpWordSerials;/* 在形状判断中为字的 */
	vector<int> wordFlag;

	vector<Point> newMaxP,newMinP;//(y,x)
	vector<Point> newPosition;
	vector<Vec3d> newColor;
	//vector<Vec3d> newMeanColorOfBoxs;
	//vector<Vec3d> newMedianColorOfBoxs;
	//vector<Vec3d> newMedianColorOfBoxs_0;
	//vector<Vec3d> newMedianColorOfBoxs_1;
	vector<int> newIsPlant;
	vector<int> newedge;
	vector<double> newedgeRatio;
	Mat clustnect;
	Mat clustnectN;
	Mat clustnectDirec;/* 标记POI水平（1）还是竖直（2） */

	/* 第一次SVM确定是字的组成的链 */
	Mat clustnect_1S;
	Mat clustnectN_1S;
	Mat clustnectDirec_1S;/* 标记POI水平（1）还是竖直（2） */

	Mat meanW;
	Mat angle;
	Mat angle_1S;

public:
	/* 显示框信息 */
	void drawBox(Mat &dst);
	void drawNewBox(Mat &dst);
	/* 合并框 */
	void mergeBox();
	void iterMerge(Mat &pwDist,Mat &pwX,Mat &pwY,Mat &colorDist,Mat &nect,Mat &nectN,int type);
	void realMerge(Mat &nect,Mat &nectN);
	/* 框聚类 */
	void clusterBox();
	/* 第一次SVM确定的字，将他们组成链 */
	void clusterBox_1s();
	void iterCluster_Hori(Mat &pwDistX,Mat &pwDistY,Mat &hzX,Mat &hzY,Mat &colorDist,Mat &nect,Mat &nectN, Mat &nectDire, double disThresh);
	void iterCluster_Vert(Mat &pwDistX,Mat &pwDistY,Mat &hzX,Mat &hzY,Mat &colorDist,Mat &nect,Mat &nectN, Mat &nectDire, double disThresh);
	void selectCluster(Mat &dst,string dir, double minification, vector<vector<Point> >& POIWordMinP, vector<vector<Point> >& POIWordMaxP);
	void selectCluster_1S(Mat &dst,string dir, double minification, vector<vector<Point> >& POIWordMinP, vector<vector<Point> >& POIWordMaxP);
	void hzangle_Hori();
	void hzangle_Vert();
	void hzangle_Hori_1s();
	void hzangle_Vert_1s();
	void hzdist();
	void hzmeanW();
	void hzsort_Hori();
	void hzsort_Vert();
	void hzsort_Hori_1s();
	void hzsort_Vert_1s();
	/* 去除植被框 */
	void eliminatePlantBox_Median(Mat &src);
	void eliminatePlantBox_Median_01(Mat &src);
	void writeTxt(Mat &dst,string dir,string name);
	/* 在部首合并之前，对符合字形状特诊的box，加一次SVM判断，若为字，在wordFlag中标记为2 */
	void firstSVMPredict(Mat src);

};

class imageHZ{
private:
	/* 图像 */
	Mat src;
	int sz;
	/* 栈 */
	Point* stack;
	int stTop;
	/* 聚类 */
	Point* cPixel;/* 记录聚类的点 */
	int clusterNum;/* 记录有多少个类 */
	int* cIdx;/* 记录每个类在cPixel的结束索引号 */
	Vec3d* clusterColor;/* 记录每个类的颜色 */
	int* cEdgeLen;
	//box
	boxHZ box;
	Mat labelImg;
	

public:
	
	imageHZ():box()
	{
		sz = 0;stTop=0;clusterNum=0;
		stack=NULL;cPixel=NULL;clusterColor=NULL;cIdx=NULL;
		printf("Without an Image,this class is useless!");
	}

	imageHZ(Mat &img):box()
	{
		src=img;
		sz = src.rows*src.cols;
		stTop=0;clusterNum=0;
		stack = new Point[sz];
		cPixel = new Point[sz];
		cIdx = new int[sz];
		clusterColor = new Vec3d[sz];
		cEdgeLen = new int[sz];
		for(int i=0;i<sz;++i) cEdgeLen[i]=0;
	}
	~imageHZ()
	{
		if(stack) delete[] stack;
		if(cPixel) delete[] cPixel;
		if(cIdx) delete[] cIdx;
		if(clusterColor) delete[] clusterColor;
		if(cEdgeLen) delete[] cEdgeLen;
	}

	/* 对新的图像进行处理，当图像尺度不超过之前处理的图像时，不需要重新分配内存 */
	void loadNewImage(Mat &img);
	/* 对图像进行聚类处理 */
	void clusterPixels(double thrpixel,double thrmean,  Mat labelResize);
	/* 把聚类展示为图像形式 */
	void cluster2Img(Mat &dst);
	void cluster2edge();
	/* 把聚类组织成box形式的信息 */
	void cluster2box();
	void selectBox(Mat &dst, string dir, double minification, 
		vector<vector<Point> >& POIWordMinP, vector<vector<Point> >& POIWordMaxP)
	{
		box.selectCluster(dst, dir, minification, POIWordMinP,POIWordMaxP);
		box.selectCluster_1S(dst, dir, minification,POIWordMinP,POIWordMaxP);
	}

	void mergeBox()
	{
		box.clusterBox_1s();
		box.mergeBox();
		box.clusterBox();
		
	}
	/* 图像滤波平滑 */
	void fastWLS(int T,double lamda,double omiga,Mat &dst);
	void fastWLS3C(int T,double lamda,double omiga,Mat &dst3);
	/* 去除植被框 */
	//void eliminatePlant_Mean(Mat &src){box.eliminatePlantBox_Mean(src);}
	void eliminatePlant_Median(Mat &src){box.eliminatePlantBox_Median(src);}
	void eliminatePlant_Median_01(Mat &src){box.eliminatePlantBox_Median_01(src);}
	void writeFile(Mat &dst, string dir, string name){box.writeTxt(dst, dir, name);}
	/* 多尺度计算后画框 */
	void drawBoxPOI(Mat &dst, string dir, string name, vector<Rect> labelRect, vector<int>& hitFlagInOneImg);
	void drawBoxPOI_SVM(Mat &dst, string dir, string name, vector<Rect> labelRect, vector<int>& hitFlagInOneImg);
	/* 第一次SVM */
	void firstSVM(Mat src){box.firstSVMPredict(src);}
	/* 第二次SVM */
	void selectBox_SVM(Mat src, string saveSampleDir);
	/* 求出两个矩形相交的部分占r1∪r2的比例 */
	double ComputeOverLapRect(Rect r1, Rect r2);	
	/* 求出两个矩形相交的部分占r1的比例 */
	double ComputeOverLapRect_r1(Rect r1, Rect r2);
	/* 找出相近框，保留位置靠后的 */
	void removeSameRect(vector<vector<Point> > minP, vector<vector<Point> > maxP, vector<vector<int> > flags,
		vector<Point>& minPRemoveSame, vector<Point>& maxPRemoveSame);
	/* 输入useSVMModel,每个框用每个模型判断一次 */
	vector<vector<int> > SVMJudege( Mat dst, vector<Point> minPRemoveSame,
		vector<Point> maxPRemoveSame, vector<Point> &IndentifyWordMaxP,
		vector<Point> &IndentifyWordMinP, vector<string> useSVMName, vector<int> SVMSerial, vector<CvSVM *> &svmModels);
	/* 将判定为字的框出来 */ 
	void drawIndentifyWord( Mat& drawWordBoxImg, vector<Point> IndentifyWordMaxP, vector<Point> IndentifyWordMinP, 
		vector<vector<int> > wordFlags, vector<string> wordMap);
	/* 判定定位框是否命中 */
	void computeLocHit(vector<Point> minPRemoveSame, vector<Point> maxPRemoveSame,
		vector<int>& locWordCata, vector<int> labelRectCata,
		vector<Rect> labelRects, int& allLabelNum, int& allLocHitNum);

	void drawBox(Mat src)
	{
		Mat dst = src.clone();
		box.drawBox(dst);
	}
	
	void drawNewBox(Mat src)
	{
		Mat dst = src.clone();
		box.drawNewBox(dst);
	}

};

