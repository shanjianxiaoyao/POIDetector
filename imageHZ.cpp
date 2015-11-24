#include <opencv2/objdetect/objdetect.hpp>

#include "imageHZ.h"

void SVMDetector::readMapFile(string mapFileName)
{
	ifstream infile(mapFileName.c_str());
	if(!infile)
	{
		cout << "open error!" << endl; 
		return;
	}
	while(!infile.eof())
	{
		string str;
		getline(infile, str);
		int ind = str.find_first_of("\t");
		wordMap.push_back(str.substr(0, ind));	
	}
}
//Windows
//{
//	ifstream infile(mapFileName);
//	if(!infile)
//	{
//		cout << "open error!" << endl; 
//		return;
//	}
//	while(!infile.eof())
//	{
//		string str;
//		getline(infile, str);
//		int ind = str.find_first_of(" ");
//		wordMap.push_back(str.substr(0, ind));	
//	}
//}


vector<int> SVMDetector::readInitialModelSerials(string fileName)
{
	vector<int> initialModelSerials;
	ifstream infile(fileName.c_str());
	if(!infile)
	{
		cout << "open error!" << endl; 
		return initialModelSerials;
	}

	int lineNum = 0;
	while(!infile.eof()/* && lineNum < 100*/)
	{
		++lineNum;
		string str;
		getline(infile, str);	
		initialModelSerials.push_back(atoi(str.c_str()));
	}
	sort(initialModelSerials.begin(), initialModelSerials.end());
	return initialModelSerials;
}

void SVMDetector::getUseModel(string imgName, vector<string>& useSVMModelNames, vector<int>& useSVMSerial,
	vector<string> &wordMap, string modelDir)
{
	const char* pstr = imgName.c_str();
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

		if( (*pstr & 0x80) == 0x80 )
		{
			buf[0] = *pstr;
			++ pstr;
			buf[1] = *pstr;
			++ pstr;
			buf[2] = *pstr;
		}else{
			buf[0] = *pstr;
			buf[1] = '\0';
		}


		for (int j = 0; j < wordMap.size(); ++j)
		{
			char cF[10];
			sprintf(cF, "%d", j);
			if ((string)buf == wordMap[j])
			{
				string modelName = modelDir + "word_" + string(cF) + ".model";
				useSVMModelNames.push_back(modelName);//添加模型Name
				useSVMSerial.push_back(j);
			}
		}
	}
}

vector<OutBlock> SVMDetector::svmDetectPOI(string poipath, string poiname)
{
	Mat srcc = imread(poipath);

	string modelDir = "/home/houkai/POI_v2/poi.conf/svm.model/";

	vector<OutBlock> objects;

	if (!srcc.data)
	{  
		return objects;
	}

	vector<string> useSVMName;
	vector<int> useSVMSerials;
	getUseModel(poiname, useSVMName, useSVMSerials, wordMap, modelDir);

    //cout << "End  getUseModel !!" << endl;
	/* 裁剪下方1/4图像 */
	Rect roi(0, 0, srcc.cols, (srcc.rows*3/4));
	Mat src_ori1(srcc, roi);
	Mat src_ori( (srcc.rows*3/4), srcc.cols, srcc.type());
	resize(src_ori1,src_ori,Size(src_ori.cols,src_ori.rows));	

	Mat labelImage(src_ori.rows,src_ori.cols,CV_8UC1);
	labelImage.setTo(255);

    // cout << " End  cut !!" << endl;
    
   // Mat src_ori;  
   // resize(srcc,src_ori,Size(srcc.cols,srcc.rows));   
   // Mat labelImage(src_ori.rows,src_ori.cols,CV_8UC1);
    //labelImage.setTo(255);

	double a[5] = {1.0, 0.8, 0.5, 0.3, 0.1}; 
	vector<double> onceMinificationV(a, a+5);
	vector<vector<Point> > POIWordMaxP, POIWordMinP;/* 字符排列完成后，POI字链中，每一个字的坐标 */
	for (int mulScaleNum = 0; mulScaleNum < 5; ++mulScaleNum)
	{
		Mat src;
		Mat labelResize;
		double minification = onceMinificationV[mulScaleNum];
		resize(src_ori,src,Size((int)src_ori.cols*minification, (int)src_ori.rows*minification));
		resize(labelImage, labelResize, Size((int)labelImage.cols*minification, (int)labelImage.rows*minification));

		imageHZ hz(src);
		Mat dst;
		//Mat gray; 
		hz.fastWLS(1,4.0,0.04,dst);

		hz.loadNewImage(dst);

		hz.clusterPixels(40*40,25*25, labelResize);

		hz.cluster2box();

		hz.mergeBox();

		hz.selectBox(src, poipath, minification, POIWordMinP, POIWordMaxP);
	}

     //cout << "End  locate !!" << endl;

	imageHZ hz(src_ori);

	/* 相近框只保留一个 */
	vector<vector<int> > flags;
	for (int f_i = 0; f_i < POIWordMinP.size(); ++f_i)
	{
		vector<int> temp(POIWordMinP[f_i].size(), 0);
		flags.push_back(temp);
	}
	if(POIWordMinP.size() == 0)
	{
		return objects;
	}
	vector<Point> minPRemoveSame;
	vector<Point> maxPRemoveSame;
	hz.removeSameRect(POIWordMinP, POIWordMaxP, flags, minPRemoveSame, maxPRemoveSame);

    // cout << "End removesame!!" << endl;

	vector<Point> IndentifyWordMaxP, IndentifyWordMinP;/* 字符排列完成后，POI字链中，每一个字的坐标 */
	vector<vector<int> > wordFlags = hz.SVMJudege( src_ori, minPRemoveSame, maxPRemoveSame,
		IndentifyWordMaxP, IndentifyWordMinP, useSVMName, useSVMSerials, svmModels);
	
    // cout << "End  svmjudge !!" << endl;
    
    //output vector<OutBlock>
	for (int i = 0; i < wordFlags.size(); ++ i)
	{
		for(int j = 0; j < wordFlags[i].size(); ++j)
		{
			Rect r(IndentifyWordMinP[i].y, IndentifyWordMinP[i].x,IndentifyWordMaxP[i].y - IndentifyWordMinP[i].y + 1,
				IndentifyWordMaxP[i].x - IndentifyWordMinP[i].x + 1);
			int label = wordFlags[i][j];
			string hz = wordMap[wordFlags[i][j]];
			objects.push_back(OutBlock(r, label, hz));
		}
	}
	return objects;

}

void SVMDetector::initialModels(string modelsDir, int partLoadFlag)
{
	//vector<CvSVM> svmModels;
	int modelNum = wordMap.size();
	for (int i = 0; i < modelNum; ++i)
	{
		CvSVM *SVMDetector = new CvSVM();
		svmModels.push_back(SVMDetector);
	}
	/* 需要一次性加载的Model序号 */
	if (partLoadFlag == 1)
	{
		string loadModelTxtName = modelsDir + "500.txt";
		vector<int> initialModelSerials = readInitialModelSerials(loadModelTxtName);
		int i_flag = 0;
		for (int j = 0; j < initialModelSerials.size(); ++j)
		{
			for (int i = i_flag; i < modelNum; ++i)
			{
				if (i == initialModelSerials[j])
				{
					char cF[10];
                    sprintf(cF, "%d", i);
					//itoa(i, cF, 10);
					string modelName = modelsDir + "word_" + string(cF) + ".model";		
					svmModels[i]->load(modelName.c_str());
					i_flag = i+1;
					break;
				}	
			}				
		}
	}
	else
	{
		for (int i = 0; i < modelNum; ++i)
		{
			char cF[10];
            sprintf(cF, "%d", i);
			//itoa(i, cF, 10);
			string modelName = modelsDir + "word_" + string(cF) + ".model";		
			svmModels[i]->load(modelName.c_str());
		}
	}
	//return svmModels;
}

/* 求出两个矩形相交的部分占r1的比例 */
double imageHZ::ComputeOverLapRect_r1(Rect r1, Rect r2)
{
	Rect r;
	/* 计算r1和r2的中心点坐标 */
	int r1EndX = r1.x+r1.width-1;
	int r1EndY = r1.y+r1.height-1;
	int r2EndX = r2.x+r2.width-1;
	int r2EndY = r2.y+r2.height-1;
	int centerR1_x = (r1EndX + r1.x) / 2;
	int centerR1_y = (r1EndY + r1.y) / 2;
	int centerR2_x = (r2EndX + r2.x) / 2;
	int centerR2_y = (r2EndY + r2.y) / 2;

	/* 两个矩形不相交 */
	if ( abs(centerR2_x - centerR1_x) > (r1.width + r2.width)/2
		|| abs(centerR2_y - centerR1_y) > (r1.height + r2.height)/2)
	{
		return 0.0;
	}
	else
	{
		r.x = (r1.x > r2.x) ? r1.x : r2.x;
		r.y = (r1.y > r2.y) ? r1.y : r2.y;
		int rEndX = (r1EndX < r2EndX) ? r1EndX : r2EndX;
		int rEndY = (r1EndY < r2EndY) ? r1EndY : r2EndY;
		r.height = rEndY - r.y + 1;
		r.width = rEndX - r.x + 1;
		return ((double)(r.height*r.width)/(r1.height*r1.width));
	}
}

/* 求出两个矩形相交的部分占r1∪r2的比例 */
double imageHZ::ComputeOverLapRect(Rect r1, Rect r2)
{
	/* 计算r1和r2的中心点坐标 */
	int r1EndX = r1.x+r1.width-1;
	int r1EndY = r1.y+r1.height-1;
	int r2EndX = r2.x+r2.width-1;
	int r2EndY = r2.y+r2.height-1;
	int centerR1_x = (r1EndX + r1.x) / 2;
	int centerR1_y = (r1EndY + r1.y) / 2;
	int centerR2_x = (r2EndX + r2.x) / 2;
	int centerR2_y = (r2EndY + r2.y) / 2;

	/* 两个矩形不相交 */
	if ( abs(centerR2_x - centerR1_x) > (r1.width + r2.width)/2
		|| abs(centerR2_y - centerR1_y) > (r1.height + r2.height)/2)
	{
		return 0.0;
	}
	else
	{
		/* 交集 */
		Rect rJ;
		rJ.x = (r1.x > r2.x) ? r1.x : r2.x;
		rJ.y = (r1.y > r2.y) ? r1.y : r2.y;
		int rJEndX = (r1EndX < r2EndX) ? r1EndX : r2EndX;
		int rJEndY = (r1EndY < r2EndY) ? r1EndY : r2EndY;
		rJ.height = rJEndY - rJ.y + 1;
		rJ.width = rJEndX - rJ.x + 1;

		/* 并集 */
		Rect rB;
		rB.x = (r1.x < r2.x) ? r1.x : r2.x;
		rB.y = (r1.y < r2.y) ? r1.y : r2.y;
		int rBEndX = (r1EndX > r2EndX) ? r1EndX : r2EndX;
		int rBEndY = (r1EndY > r2EndY) ? r1EndY : r2EndY;
		rB.height = rBEndY - rB.y + 1;
		rB.width = rBEndX - rB.x + 1;

		return ((double)(rJ.height*rJ.width)/(rB.height*rB.width));
	}
}  

cv::Mat cvtVec2Mat(const std::vector<float> & vec){
	cv::Mat fea ;
	fea.create(1, static_cast<int>( vec.size() ), CV_32F);
	cv::Mat_<float>::iterator it  = fea.begin<float>();
	cv::Mat_<float>::iterator end = fea.end<float>();
	size_t idx = 0;
	size_t idx_end = vec.size();
	for( idx =0;
		idx < idx_end && it != end;
		++idx, ++it)
	{
		(*it) = vec[idx];
	}
	return fea;
}


vector<vector<int> > imageHZ::SVMJudege( Mat dst, vector<Point> minPRemoveSame,
	vector<Point> maxPRemoveSame, vector<Point> &IndentifyWordMaxP,	vector<Point> &IndentifyWordMinP,
	vector<string> useSVMName, vector<int> SVMSerial, vector<CvSVM *> &svmModels)
{
	vector<vector<int> > wordFlags;
	for (int i = 0; i < minPRemoveSame.size(); ++i)
	{
		vector<int> oneRectFlags;
		wordFlags.push_back(oneRectFlags);
	}
	for (int i_SVM = 0; i_SVM < useSVMName.size(); ++i_SVM)
	{	
		CvSVM SVMDetector;
		/* 没有该模型 */
		if (svmModels[SVMSerial[i_SVM]]->get_support_vector_count() == 0)
		{
			char* curModelName = (char*)useSVMName[i_SVM].data();
			SVMDetector.load(curModelName);
			if (SVMDetector.get_support_vector_count() == 0)
			{
				//cout << "No" + string(useSVMName[i_SVM]) + "!!" << endl;
				continue;
			}
		}

		for (int i = 0; i < minPRemoveSame.size(); ++i)
		{
			Rect r(minPRemoveSame[i].y,minPRemoveSame[i].x,
				maxPRemoveSame[i].y-minPRemoveSame[i].y+1, maxPRemoveSame[i].x-minPRemoveSame[i].x+1);
			Mat srcROI( dst,r );
			Mat srcROIR(32, 32, srcROI.type());
			resize(srcROI, srcROIR, cv::Size(srcROIR.cols, srcROIR.rows));
			/* HOG+SVM检测 */
			vector<float>  descriptors;
			HOGDescriptor *hog = new HOGDescriptor(cv::Size(32,32), cv::Size(16,16),cv::Size(8,8), cv::Size(4,4), 9 ); /**< 特征计算器 */
			hog->compute(srcROIR, descriptors); 
			Mat feather = cvtVec2Mat(descriptors);
            delete hog;
            hog = 0;
			float response = 0;
			if(svmModels[SVMSerial[i_SVM]]->get_support_vector_count() == 0)
			{
				response= SVMDetector.predict(feather);
			}
			else
			{
				response= svmModels[SVMSerial[i_SVM]]->predict(feather);
			}
			if (response == 1)
			{
				wordFlags[i].push_back(SVMSerial[i_SVM]);
			}
			else
			{	
				/* 扩大一圈检测 */
				int boxHeight = maxPRemoveSame[i].y-minPRemoveSame[i].y+1;
				int boxWidth = maxPRemoveSame[i].x-minPRemoveSame[i].x+1;
				int addH = (boxHeight*0.05 > 3) ? boxHeight*0.05 : 3;
				int addW = (boxWidth*0.05 > 3) ? boxWidth*0.05 : 3;

				/* 越界判断 */
				int startY_add = (minPRemoveSame[i].y-addH > 0) ? minPRemoveSame[i].y-addH : 0;
				int startX_add = (minPRemoveSame[i].x-addW > 0) ? minPRemoveSame[i].x-addW : 0;
				int endY_add = (maxPRemoveSame[i].y + addH < dst.cols-1) ? (maxPRemoveSame[i].y + addH) : (dst.cols-1);
				int endX_add = (maxPRemoveSame[i].x + addW < dst.rows-1) ? (maxPRemoveSame[i].x + addW) : (dst.rows-1);
				Rect r_add(startY_add, startX_add, endY_add-startY_add+1, endX_add-startX_add+1);
				Mat srcROI_add( dst,r_add );
				Mat srcROIR_add(32, 32, srcROI_add.type());
				resize(srcROI_add, srcROIR_add, cv::Size(srcROIR_add.cols, srcROIR_add.rows));

				vector<float>  descriptors_add;
				HOGDescriptor *hog_add = new HOGDescriptor(cv::Size(32,32), cv::Size(16,16),cv::Size(8,8), cv::Size(4,4), 9 ); /**< 特征计算器 */
				hog_add->compute(srcROIR_add, descriptors_add); 
				Mat feather_add = cvtVec2Mat(descriptors_add);
                delete hog_add;
                hog_add = 0;
				float response_add = 0;
				if (svmModels[SVMSerial[i_SVM]]->get_support_vector_count() == 0)
				{
					response_add= SVMDetector.predict(feather_add);
				}
				else
				{
					response_add= svmModels[SVMSerial[i_SVM]]->predict(feather_add);
				}
				if (response_add == 1)
				{
					wordFlags[i].push_back(SVMSerial[i_SVM]);
				}
			}
		}
	}
	/* 只保留认定为字的Rect */
	vector<vector<int> > wordFlags2;
	for (int i = 0; i < minPRemoveSame.size(); ++i)
	{
		if (wordFlags[i].size() != 0)
		{
			IndentifyWordMaxP.push_back(maxPRemoveSame[i]);
			IndentifyWordMinP.push_back(minPRemoveSame[i]);
			wordFlags2.push_back(wordFlags[i]);
		}
	}
	return wordFlags2;
}

void imageHZ::removeSameRect(vector<vector<Point> > minP, vector<vector<Point> > maxP, vector<vector<int> > flags,
	vector<Point>& minPRemoveSame, vector<Point>& maxPRemoveSame)
{
	vector<Point> minPNC;
	vector<Point> maxPNC;
    //cout << "Rstart1" << endl;
	for (int i = 0; i < minP.size(); ++i)
	{
		for (int j = 0; j < minP[i].size(); ++j)
		{
			if (flags[i][j] == 0)
			{
				minPNC.push_back(minP[i][j]);
				maxPNC.push_back(maxP[i][j]);
			}			
		}
	}
    //cout << "Rstart2" << endl;
	vector<int>flag(minPNC.size(), 0);/* 1表示被合并 */
   // cout << "Rstart2_" << endl;
   // cout << minPNC.size() << endl;
    for (int i = 0; i < minPNC.size()-1; ++i)
	{
		for (int j = i+1; j < minPNC.size(); ++j)
		{
           // cout << "i" << i << "j" << j << endl;
			Rect ri(minPNC[i].y, minPNC[i].x, maxPNC[i].y-minPNC[i].y+1, maxPNC[i].x-minPNC[i].x+1 );
			Rect rj(minPNC[j].y, minPNC[j].x, maxPNC[j].y-minPNC[j].y+1, maxPNC[j].x-minPNC[j].x+1 );
			double areaPort = ComputeOverLapRect(ri, rj);
			if (areaPort > 0.8)
			{
				flag[i] = 1;
				break;
			}
		}
		if (flag[i] == 1)
		{
			continue;
		}
	}
   // cout << "Rstart3" << endl;
	for (int i = 0; i < flag.size(); ++i)
	{
		if (flag[i] == 0)
		{
			minPRemoveSame.push_back(minPNC[i]);
			maxPRemoveSame.push_back(maxPNC[i]);
		}
	}
}

void imageHZ::drawIndentifyWord( Mat& drawWordBoxImg, vector<Point> IndentifyWordMaxP, 
	vector<Point> IndentifyWordMinP, vector<vector<int> > wordFlags, vector<string> wordMap)
{
	for (int i = 0; i < IndentifyWordMaxP.size(); ++i)
	{
		rectangle(drawWordBoxImg,Point(IndentifyWordMinP[i].y, IndentifyWordMinP[i].x),
			Point(IndentifyWordMaxP[i].y, IndentifyWordMaxP[i].x),cvScalar(0,0,255,0),2,8,0);
		for (int j = 0; j < wordFlags[i].size(); ++j)
		{
			char cF[10];
            sprintf(cF, "%d", wordFlags[i][j]);
			//itoa(wordFlags[i][j], cF, 10);
			putText(drawWordBoxImg, string(cF),Point(IndentifyWordMaxP[i].y/2+IndentifyWordMinP[i].y/2, 
				IndentifyWordMaxP[i].x/2+IndentifyWordMinP[i].x/2),1, 1.0,Scalar(255,255,255),1,8,false);
		}
	}
}

void imageHZ::loadNewImage(Mat &img)
{
	src=img;
	int temp = src.rows*src.cols;
	if(temp>sz)
	{
		sz=temp;
		if(stack) delete[] stack;
		if(cPixel) delete[] cPixel;
		if(cIdx) delete[] cIdx;
		if(clusterColor) delete[] clusterColor;
		stTop=0;
		stack = new Point[sz];
		cPixel = new Point[sz];
		cIdx = new int[sz];
		clusterColor = new Vec3d[sz];
	}
}

/* 图像聚类，坐标信息保存在cPixel，cIdx，颜色信息保存在clusterColor */
void imageHZ::clusterPixels(double thrpixel,double thrmean, Mat labelResize)
{	
	//Mat isClustered = Mat::zeros(src.size(),CV_32S);//isClustered = isClustered*0;
	//for(int i=0;i<isClustered.rows;i++)
	//{
	//	isClustered.at<int>(i,0)=isClustered.at<int>(i,isClustered.cols-1)=1;
	//}
	//for(int i=0;i<isClustered.cols;i++)
	//{
	//	isClustered.at<int>(0,i)=isClustered.at<int>(isClustered.rows-1,i)=1;
	//}

	for(int i=0;i<labelResize.rows;i++)
	{
		labelResize.at<uchar>(i,0)=labelResize.at<uchar>(i,labelResize.cols-1)=125;
	}
	for(int i=0;i<labelResize.cols;i++)
	{
		labelResize.at<uchar>(0,i)=labelResize.at<uchar>(labelResize.rows-1,i)=125;
	}
	
	int pixelNum;
	int pIdx=0;
	double distP,distC;	
	Vec3d meanColor = Vec3d(0,0,0);
	Point Nb[]={Point(-1,-1),Point(-1,1),Point(1,-1),Point(1,1),Point(-1,0),Point(1,0),Point(0,-1),Point(0,1)};

	stTop=0;clusterNum=0;
	for(int i=1;i<src.rows-1;i++)
	{ 
		for(int j=1;j<src.cols-1;j++)
		{
			if(labelResize.at<uchar>(i,j)==255)
			{				
				stack[++stTop]=Point(i,j);// go into the stack;	
				labelResize.at<uchar>(i,j) = 125;
				pixelNum=0;				
				meanColor = Vec3d(0.0,0.0,0.0);

				while(stTop)
				{
					Point p = stack[stTop--];// go out of the stack;
					meanColor = (meanColor*pixelNum+(Vec3d)src.at<Vec3b>(p.x,p.y))/(pixelNum+1);
					pixelNum++;
					cPixel[pIdx++] = p;//if(pIdx>src.rows*src.cols){return;}
					
					for(int ni=0;ni<8;ni++)//search 8-connected neighbors
					{
						Point q = p+Nb[ni];
						if(labelResize.at<uchar>(q.x,q.y)==255)
						{
							Vec3d tempV = (Vec3d)src.at<Vec3b>(p.x,p.y)-(Vec3d)src.at<Vec3b>(q.x,q.y);
							distP = (tempV[0]*tempV[0]+tempV[1]*tempV[1]+tempV[2]*tempV[2])/3;
							tempV = meanColor-(Vec3d)src.at<Vec3b>(q.x,q.y);
							distC = (tempV[0]*tempV[0]+tempV[1]*tempV[1]+tempV[2]*tempV[2])/3;
							if(distP<thrpixel&&distC<thrmean)
							{
								stack[++stTop]=q;
								labelResize.at<uchar>(q.x,q.y)=125;
							}
						}
					}
				}
				clusterColor[clusterNum++] = meanColor;
				cIdx[clusterNum] = pIdx;
								
			}
		}
	}
	cIdx[0]=0;
	cluster2edge();
}

/* 聚类用图像的方式显示 */
void imageHZ::cluster2Img(Mat &dst)
{
	dst = src.clone();
	int idx=0;
	for(int i=0;i<clusterNum;i++)
	{
		for(int j=cIdx[i];j<cIdx[i+1];j++)
		{
			dst.at<Vec3b>(cPixel[j].x,cPixel[j].y)=(Vec3b)clusterColor[i];
		}
	}
}

/* 记录每个类的边缘长度 */
void imageHZ::cluster2edge()
{
	Mat label = Mat::zeros(src.rows,src.cols,CV_32S);
	for(int i=0;i<clusterNum;++i)
	{
		for(int j=cIdx[i];j<cIdx[i+1];++j)
			label.at<int>(cPixel[j].x,cPixel[j].y)=i;
	}
	for(int i=1;i<label.rows-1;++i)
		for(int j=1;j<label.cols-1;++j)
		{
			if (label.at<int>(i,j)!=label.at<int>(i-1,j-1)||\
				label.at<int>(i,j)!=label.at<int>(i-1,j)||\
				label.at<int>(i,j)!=label.at<int>(i-1,j+1)||\
				label.at<int>(i,j)!=label.at<int>(i,j-1)||\
				label.at<int>(i,j)!=label.at<int>(i,j+1)||\
				label.at<int>(i,j)!=label.at<int>(i+1,j-1)||\
				label.at<int>(i,j)!=label.at<int>(i+1,j)||\
				label.at<int>(i,j)!=label.at<int>(i+1,j+1))
			{
				cEdgeLen[label.at<int>(i,j)]++;
			}
		}
	
}

/* 对图像进行滤波 */
void imageHZ::fastWLS(int T,double lamda,double omiga,Mat &dst3)
{
	int H = src.rows;
	int W = src.cols;
	dst3 = Mat::zeros(H,W,CV_8UC3);
	Mat dst = Mat::zeros(H,W,CV_64F);
	for(int c=0;c<3;++c)
	{
		//
		for(int i=0;i<H;++i)
			for(int j=0;j<W;++j)
				dst.at<double>(i,j)=((double)src.at<Vec3b>(i,j)[c])/255;
	
		//
		for(int t=1;t<=T;++t)
		{
			double lt = 1.5*lamda*pow(4.0,T-t)/(pow(4.0,T)-1);
			Mat cx_ = Mat::zeros(H,1,CV_64F);
			Mat f_ = Mat::zeros(H,1,CV_64F);


			//1D H
			for(int j=0;j<W;++j)
			{
				Mat fh = Mat::zeros(H,1,CV_64F);
				for(int k=0;k<H;++k)
					fh.at<double>(k,0) = dst.at<double>(k,j);
				Mat gh = fh.clone();
				Mat wx1 = Mat::zeros(H,1,CV_64F);
				Mat wx2 = Mat::zeros(H,1,CV_64F);
				for(int k=1;k<H-1;++k)
				{
					wx1.at<double>(k,0)=exp(-abs(gh.at<double>(k,0)-gh.at<double>(k-1,0))/omiga);
					wx2.at<double>(k,0)=exp(-abs(gh.at<double>(k,0)-gh.at<double>(k+1,0))/omiga);
				}
				wx1.at<double>(H-1,0)=exp(-abs(gh.at<double>(H-1,0)-gh.at<double>(H-2,0))/omiga);
				wx2.at<double>(0,0)=exp(-abs(gh.at<double>(0,0)-gh.at<double>(1,0))/omiga);
				Mat ax = -lt*wx1;
				Mat cx = -lt*wx2;
				Mat bx = 1+lt*(wx1+wx2);
				cx_.at<double>(0,0)=cx.at<double>(0,0)/bx.at<double>(0,0);
				f_.at<double>(0,0)=fh.at<double>(0,0)/bx.at<double>(0,0);
				for(int k=1;k<H;++k)
				{
					double de = bx.at<double>(k,0)-cx_.at<double>(k-1,0)*ax.at<double>(k,0);
					cx_.at<double>(k,0) = cx.at<double>(k,0)/de;
					f_.at<double>(k,0) = (fh.at<double>(k,0)-f_.at<double>(k-1,0)*ax.at<double>(k,0))/de;
				}
				dst.at<double>(H-1,j)=f_.at<double>(H-1,0);
				for(int k=H-2;k>0;--k)
				{
					dst.at<double>(k,j)=f_.at<double>(k,0)-cx_.at<double>(k,0)*dst.at<double>(k+1,j);
				}
			}

			cx_ = Mat::zeros(1,W,CV_64F);f_ = Mat::zeros(1,W,CV_64F);

			//1D W
			for(int i=0;i<H;++i)
			{
				Mat fv = Mat::zeros(1,W,CV_64F);
				for(int k=0;k<W;++k) fv.at<double>(0,k) = dst.at<double>(i,k);
				Mat gv = fv.clone();
				Mat wx1 = Mat::zeros(1,W,CV_64F);Mat wx2 = Mat::zeros(1,W,CV_64F);
				for(int k=1;k<W-1;++k)
				{
					wx1.at<double>(0,k)=exp(-abs(gv.at<double>(0,k)-gv.at<double>(0,k-1))/omiga);
					wx2.at<double>(0,k)=exp(-abs(gv.at<double>(0,k)-gv.at<double>(0,k+1))/omiga);
				}
				wx1.at<double>(0,W-1)=exp(-abs(gv.at<double>(0,W-1)-gv.at<double>(0,W-2))/omiga);
				wx2.at<Vec3d>(0,0)=exp(-abs(gv.at<double>(0,0)-gv.at<double>(0,1))/omiga);
				Mat ax = -lt*wx1;
				Mat cx = -lt*wx2;
				Mat bx = 1+lt*(wx1+wx2);
				cx_.at<double>(0,0)=cx.at<double>(0,0)/bx.at<double>(0,0);
				f_.at<double>(0,0)=fv.at<double>(0,0)/bx.at<double>(0,0);
				for(int k=1;k<W;++k)
				{
					double de = bx.at<double>(0,k)-cx_.at<double>(0,k-1)*ax.at<double>(0,k);
					cx_.at<double>(0,k) = cx.at<double>(0,k)/de;
					f_.at<double>(0,k) = (fv.at<double>(0,k)-f_.at<double>(0,k-1)*ax.at<double>(0,k))/de;
				}
				dst.at<double>(i,W-1)=f_.at<double>(0,W-1);
				for(int k=W-2;k>0;--k)
				{
					dst.at<double>(i,k)=f_.at<double>(0,k)-cx_.at<double>(0,k)*dst.at<double>(i,k+1);
				}
			}

		}
		for(int i=0;i<H;++i) for(int j=0;j<W;++j) dst3.at<Vec3b>(i,j)[c]=(uchar)(dst.at<double>(i,j)*255.0);
	}

}

/* 把聚类区域简化成box */
void imageHZ::cluster2box()
{
	int idx=0;
	Point MaxP,MinP;
	for(int i=0;i<clusterNum;++i)
	{
		int num = cIdx[i+1]-cIdx[i];
		if(num>5&&num<15000)
		{
			MaxP = Point(0,0);MinP = Point(100000,100000);
			for(int j=cIdx[i];j<cIdx[i+1];++j)
			{
				if(cPixel[j].x>MaxP.x) MaxP.x=cPixel[j].x;
				if(cPixel[j].x<MinP.x) MinP.x=cPixel[j].x;
				if(cPixel[j].y>MaxP.y) MaxP.y=cPixel[j].y;
				if(cPixel[j].y<MinP.y) MinP.y=cPixel[j].y;
			}
			double xx = MaxP.x-MinP.x;
			double yy = MaxP.y-MinP.y;
			double ratio = xx/yy;
			double boxsz = xx*yy;
			double wrt = double(num)/double(boxsz);
			double el = cEdgeLen[i] / ((xx+yy)*2);
			//if(ratio<2&&ratio>0.5&&xx>5&&yy>5&&xx<200&&yy<200&&wrt>0.1)
			if(ratio<3&&ratio>0.33&&xx>3&&yy>3&&xx<300&&yy<300&&wrt>0.1)
			{
				box.vMaxP.push_back(MaxP);box.vMinP.push_back(MinP);/* box的两个点坐标 */
				box.Color.push_back((Vec3b)clusterColor[i]);/* box的颜色 */
				box.xyRatio.push_back(ratio);/* box的长宽比例 */
				box.Region.push_back(num);/* box代表的类的区域面积 */
				box.Size.push_back(boxsz);/* box的面积 */
				box.Position.push_back(Point((MaxP.x+MinP.x)/2,(MaxP.y+MinP.y)/2));/* box的中心位置 */
				box.xlen.push_back(xx);
				box.ylen.push_back(yy);
				box.edge.push_back(cEdgeLen[i]);/* Region的边缘长度和box的边缘长度之比 */
				box.edgeRatio.push_back(el);
				if(ratio<1.3&&ratio>0.7)
				{
					box.sharpWordSerials.push_back(box.vMaxP.size()-1);
				}			
			}
			
		}
	}
}

void boxHZ::drawBox(Mat &dst)
{
	for(int i=0;i<vMaxP.size();i++)
	{
		char areaSerialC[100];
        sprintf(areaSerialC, "%d", i);
		//itoa(i,areaSerialC, 10); 
		putText(dst,areaSerialC,Point(vMaxP[i].y,vMaxP[i].x),1,1.0,Scalar(255,255,255),1,8,false);/* 在图片中输出字符 */   
		rectangle(dst,Point(vMinP[i].y,vMinP[i].x),Point(vMaxP[i].y,vMaxP[i].x),cvScalar(0,0,255),1,8,0);
	}
	imwrite("clusterBox.png", dst);
}

void boxHZ::drawNewBox(Mat &dst)
{
	std::cout<<"newMaxP.size(): "<<newMaxP.size()<<std::endl;
	for(int i=0;i<newMaxP.size();i++)
	{
		char areaSerialC[100];
        sprintf(areaSerialC, "%d", i);
		//itoa(i,areaSerialC, 10); 
		putText(dst,areaSerialC,Point(newMaxP[i].y,newMaxP[i].x),1,1.0,Scalar(255,255,255),1,8,false);/* 在图片中输出字符 */   
		rectangle(dst,Point(newMinP[i].y,newMinP[i].x),Point(newMaxP[i].y,newMaxP[i].x),cvScalar(0,0,255),1,8,0);//cvScalar(newColor[i][0],newColor[i][1],newColor[i][2])
	}
	imwrite("mergeBox.png", dst);
}

/* 因为汉字的部首分离，合并box */
void boxHZ::mergeBox()
{
	/*std::cout<<"vMaxP.size(): "<<vMaxP.size()<<std::endl;*/
	int Num = vMaxP.size();
	Mat pwDist = Mat::zeros(Num,Num,CV_64F);
	Mat pwX = Mat::zeros(Num,Num,CV_32F);
	Mat pwY = Mat::zeros(Num,Num,CV_32F);
	Mat colorDist = Mat::zeros(Num,Num,CV_64F);
	Vec3d cdt;
	int dx,dy,dcx,dcy;
	for(int i=0;i<Num-1;++i)
	{
		for(int j=i+1;j<Num;++j)
		{
			cdt = Color[i]-Color[j];
			colorDist.at<double>(i,j) = colorDist.at<double>(j,i) = cdt[0]*cdt[0]+cdt[1]*cdt[1]+cdt[2]*cdt[2];
			dcx = abs(Position[i].x-Position[j].x);
			dcy = abs(Position[i].y-Position[j].y);
			dx = dcx-(vMaxP[i].x+vMaxP[j].x-vMinP[i].x-vMinP[j].x)/2;
			dy = dcy-(vMaxP[i].y+vMaxP[j].y-vMinP[i].y-vMinP[j].y)/2;
			dx = dx<0?0:dx;
			dy = dy<0?0:dy;
			pwDist.at<double>(i,j) = pwDist.at<double>(j,i) = pow((dx*dx+dy*dy),0.5);
			pwX.at<int>(i,j) = dx;
			pwY.at<int>(i,j) = dy;
		}
	}
	Mat nect = Mat::zeros(Num,50,CV_32S);
	for (int i=0;i<Num;++i) {nect.at<int>(i,0)=i;}//nect
	Mat nectN = Mat::ones(Num,1,CV_32S);
	/*std::cout<<"nectN.rows:  "<<nectN.rows<<std::endl;*/

	for(int i=0;i<50;++i)
	{
		int nn=0;for(int k=0;k<nectN.rows;++k) nn+=nectN.at<int>(k,0);	
		iterMerge(pwDist,pwX,pwY,colorDist,nect,nectN,1);
		int mm=0;for(int k=0;k<nectN.rows;++k) mm+=nectN.at<int>(k,0);
		if(mm==nn) break;
	}

	/*	for(int i=0;i<10;++i)
	{
		int nn=0;for(int k=0;k<nectN.rows;++k) nn+=nectN.at<int>(k,0);	
		iterMerge(pwDist,pwX,pwY,colorDist,nect,nectN,2);
		int mm=0;for(int k=0;k<nectN.rows;++k) mm+=nectN.at<int>(k,0);
		if(mm==nn) break;
	}*/

	realMerge(nect,nectN);
}

/* 合并的目的是为了得到更正的box,所以策略的选择也很简单--如果足够正了，就不用合并 */
/* 1.大小相当的box,合并获得更好的box,否则距离为0也不用合并 */
/* 2.大小不等的box,距离为0则合并 */
/* 3.其它情况，获得更好的box */
void boxHZ::iterMerge(Mat &pwDist,Mat &pwX,Mat &pwY,Mat &colorDist,Mat &nect,Mat &nectN,int type)
{
	int connect;
	for(int i=nectN.rows-1;i>0;--i)
	{/*std::cout<<"nect: "<<nect.row(i)<<std::endl;*/
		if(nectN.at<int>(i,0))
		{
			for(int j=i-1;j>=0;--j)
			{
				if(nectN.at<int>(j,0))
				{
					connect = 0;
					for(int ki=0;ki<nectN.at<int>(i,0);++ki)
						for(int kj=0;kj<nectN.at<int>(j,0);++kj)
						{
							/*if(colorDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<1000&&\
								((pwDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<1)||\
								(pwDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<5&&\
								
								))\

								)*/
							int ii = nect.at<int>(i,ki);
							int jj = nect.at<int>(j,kj);
							if(colorDist.at<double>(ii,jj)<500)//颜色
							{
								int mxx = max(vMaxP[ii].x,vMaxP[jj].x);
								int mxy = max(vMaxP[ii].y,vMaxP[jj].y);
								int mnx = min(vMinP[ii].x,vMinP[jj].x);
								int mny = min(vMinP[ii].y,vMinP[jj].y);
								int xa = vMaxP[ii].x-vMinP[ii].x;
								int ya = vMaxP[ii].y-vMinP[ii].y;
								int xb = vMaxP[jj].x-vMinP[jj].x;
								int yb = vMaxP[jj].y-vMinP[jj].y;
								double da = abs(xyRatio[ii]-1);
								double db = abs(xyRatio[jj]-1);
								double dab = abs(double(mxx-mnx)/double(mxy-mny)-1);
								double szr = Size[ii]/Size[jj];
								if(type==1&&pwDist.at<double>(ii,jj)<1)
								{
									connect = 1;
								}else if(type==2&&pwDist.at<double>(ii,jj)<1&&dab<0.2)
								{
									connect = 1;
								}
							}
						}
					if(connect)
					{
						for(int ki=0;ki<nectN.at<int>(i,0);++ki)
						{
							nect.at<int>(j,nectN.at<int>(j,0)++)=nect.at<int>(i,ki);
							nect.at<int>(i,ki)=0;
						}
						nectN.at<int>(i,0)=0;
						/*std::cout<<"nect: "<<nect.row(i)<<std::endl;*/
						break;
					}
				}
			}
		}
	}
}

void boxHZ::realMerge(Mat &nect,Mat &nectN)
{
	int stTop = 0;
	for(int i=0;i<nectN.rows;++i)
	{
		if(nectN.at<int>(i,0))
		{
			Point mxP = Point(0,0);
			Point mnP = Point(100000,100000);
			Vec3d ncl = Vec3d(0.0,0.0,0.0);
			int el = 0;
			int k;
			for(k=0;k<nectN.at<int>(i,0);++k)
			{
				int id = nect.at<int>(i,k);
				mxP.x = max(mxP.x,vMaxP[id].x);
				mxP.y = max(mxP.y,vMaxP[id].y);
				mnP.x = min(mnP.x,vMinP[id].x);
				mnP.y = min(mnP.y,vMinP[id].y);
				ncl+=Color[id];
				el+= edge[id];
			}
			double lx = mxP.x-mnP.x;
			double ly = mxP.y-mnP.y;
			double xyrio = lx/ly;
			double eratio = double(el)/((lx+ly)*2);
			if(1)//(xyrio>0.4&&xyrio<2.5&&lx>8&&lx<200&&ly>8&&ly<200&&eratio>0.7)
			{
				newMaxP.push_back(mxP);
				newMinP.push_back(mnP);
				newPosition.push_back(Point((mxP.x+mnP.x)/2,(mxP.y+mnP.y)/2));
				for(int c=0;c<3;c++) ncl[c]= ncl[c]/k;				
				newColor.push_back(ncl);
				newedge.push_back(el);
				newedgeRatio.push_back(eratio);
				stTop++;
			}
		}
	}
	/*std::cout<<"newMaxP.size(): "<<newMaxP.size()<<std::endl;*/
	/*int nnnn=0;
	for(int k=0;k<nectN.rows;k++) nnnn+=nectN.at<int>(k,0);*/
	/*std::cout<<"nnnn:  "<<nnnn<<std::endl;*/
}

void boxHZ::clusterBox()
{
	int Num = newMaxP.size();
	Mat hzX = Mat::zeros(Num,1,CV_32S);
	Mat hzY = Mat::zeros(Num,1,CV_32S);
	Mat pwDistX = Mat::zeros(Num,Num,CV_64F);
	Mat pwDistY = Mat::zeros(Num,Num,CV_64F);
	Mat colorDist = Mat::zeros(Num,Num,CV_64F);
	for(int i=0;i<Num-1;++i)
	{
		hzX.at<int>(i,0) = newMaxP[i].x-newMinP[i].x;
		hzY.at<int>(i,0) = newMaxP[i].y-newMinP[i].y;
	}
	Vec3d cdt;
	int dx,dy,dcx,dcy;
	for(int i=0;i<Num-1;++i)
	{
		for(int j=i+1;j<Num;++j)
		{
			cdt = newColor[i]-newColor[j];
			colorDist.at<double>(i,j) = colorDist.at<double>(j,i) = cdt[0]*cdt[0]+cdt[1]*cdt[1]+cdt[2]*cdt[2];
			dcx = abs(newPosition[i].x-newPosition[j].x);//abs(newMaxP[i].x+newMinP[i].x-newMaxP[j].x-newMinP[j].x)/2;
			dcy = abs(newPosition[i].y-newPosition[j].y);//abs(newMaxP[i].y+newMinP[i].y-newMaxP[j].y-newMinP[j].y)/2;
			dx = dcx-(hzX.at<int>(i,0)+hzX.at<int>(j,0))/2;//(newMaxP[i].x+newMaxP[j].x-newMinP[i].x-newMinP[j].x)/2;
			dy = dcy-(hzY.at<int>(i,0)+hzY.at<int>(j,0))/2;//(newMaxP[i].y+newMaxP[j].y-newMinP[i].y-newMinP[j].y)/2;
			/*dx = dx<0?0:dx;
			dy = dy<0?0:dy;*/
			pwDistX.at<double>(i,j) = pwDistX.at<double>(j,i) = dx;//<0?0:dx;
			pwDistY.at<double>(i,j) = pwDistY.at<double>(j,i) = dy;//<0?0:dy;

		}
	}
	/*Mat nect = Mat::zeros(Num,50,CV_32S);
	for (int i=0;i<Num;++i) {nect.at<int>(i,0)=i;}
	Mat nectN = Mat::ones(Num,1,CV_32S);*/
	clustnect = Mat::zeros(Num,50,CV_32S);
	for (int i=0;i<Num;++i) {clustnect.at<int>(i,0)=i;}
	clustnectN = Mat::ones(Num,1,CV_32S);
	clustnectDirec = Mat::zeros(Num,1,CV_32S);

	for(int i=0;i<50;++i)
	{
		int nn=0;for(int k=0;k<clustnectN.rows;++k) nn+=clustnectN.at<int>(k,0);	
		iterCluster_Hori(pwDistX,pwDistY,hzX,hzY,colorDist,clustnect,clustnectN,clustnectDirec,4);
		int mm=0;for(int k=0;k<clustnectN.rows;++k) mm+=clustnectN.at<int>(k,0);
		if(mm==nn) break;
	}
	hzsort_Hori();
	hzangle_Hori();

	for(int i=0;i<50;++i)
	{
		int nn=0;for(int k=0;k<clustnectN.rows;++k) nn+=clustnectN.at<int>(k,0);	
		iterCluster_Vert(pwDistX,pwDistY,hzX,hzY,colorDist,clustnect,clustnectN, clustnectDirec,4);
		int mm=0;for(int k=0;k<clustnectN.rows;++k) mm+=clustnectN.at<int>(k,0);
		if(mm==nn) break;
	}
	hzsort_Vert();
	hzangle_Vert();
}

void boxHZ::clusterBox_1s()
{
	int Num = sharpWordSerials.size();
	Mat hzX = Mat::zeros(Num,1,CV_32S);
	Mat hzY = Mat::zeros(Num,1,CV_32S);
	Mat pwDistX = Mat::zeros(Num,Num,CV_64F);
	Mat pwDistY = Mat::zeros(Num,Num,CV_64F);
	Mat colorDist = Mat::zeros(Num,Num,CV_64F);
	for(int i=0;i<Num-1;++i)
	{
		hzX.at<int>(i,0) = vMaxP[sharpWordSerials[i]].x-vMinP[sharpWordSerials[i]].x;
		hzY.at<int>(i,0) = vMaxP[sharpWordSerials[i]].y-vMinP[sharpWordSerials[i]].y;
	}
	Vec3d cdt;
	int dx,dy,dcx,dcy;
	for(int i=0;i<Num-1;++i)
	{
		int si = sharpWordSerials[i];
		for(int j=i+1;j<Num;++j)
		{
			int sj = sharpWordSerials[j];
			cdt = Color[sharpWordSerials[i]]-Color[sharpWordSerials[j]];
			colorDist.at<double>(i,j) = colorDist.at<double>(j,i) = cdt[0]*cdt[0]+cdt[1]*cdt[1]+cdt[2]*cdt[2];
			dcx = abs(Position[sharpWordSerials[i]].x-Position[sharpWordSerials[j]].x);//abs(newMaxP[i].x+newMinP[i].x-newMaxP[j].x-newMinP[j].x)/2;
			dcy = abs(Position[sharpWordSerials[i]].y-Position[sharpWordSerials[j]].y);//abs(newMaxP[i].y+newMinP[i].y-newMaxP[j].y-newMinP[j].y)/2;
			dx = dcx-(hzX.at<int>(i,0)+hzX.at<int>(j,0))/2;//(newMaxP[i].x+newMaxP[j].x-newMinP[i].x-newMinP[j].x)/2;
			dy = dcy-(hzY.at<int>(i,0)+hzY.at<int>(j,0))/2;//(newMaxP[i].y+newMaxP[j].y-newMinP[i].y-newMinP[j].y)/2;
			//dx = dx<0?0:dx;
			//dy = dy<0?0:dy;
			pwDistX.at<double>(i,j) = pwDistX.at<double>(j,i) = dx;//<0?0:dx;
			pwDistY.at<double>(i,j) = pwDistY.at<double>(j,i) = dy;//<0?0:dy;

		}
	}
	/*Mat nect = Mat::zeros(Num,50,CV_32S);
	for (int i=0;i<Num;++i) {nect.at<int>(i,0)=i;}
	Mat nectN = Mat::ones(Num,1,CV_32S);*/
	clustnect_1S = Mat::zeros(Num,50,CV_32S);
	for (int i=0;i<Num;++i) {clustnect_1S.at<int>(i,0)=i;}
	clustnectN_1S = Mat::ones(Num,1,CV_32S);
	clustnectDirec_1S = Mat::zeros(Num,1,CV_32S);

	for(int i=0;i<50;++i)
	{
		int nn=0;for(int k=0;k<clustnectN_1S.rows;++k) nn+=clustnectN_1S.at<int>(k,0);	
		iterCluster_Hori(pwDistX,pwDistY,hzX,hzY,colorDist,clustnect_1S,clustnectN_1S,clustnectDirec_1S,-4);
		int mm=0;for(int k=0;k<clustnectN_1S.rows;++k) mm+=clustnectN_1S.at<int>(k,0);
		if(mm==nn) break;
	}
	hzsort_Hori_1s();
	hzangle_Hori_1s();

	for(int i=0;i<50;++i)
	{
		int nn=0;for(int k=0;k<clustnectN_1S.rows;++k) nn+=clustnectN_1S.at<int>(k,0);	
		iterCluster_Vert(pwDistX,pwDistY,hzX,hzY,colorDist,clustnect_1S,clustnectN_1S, clustnectDirec_1S, -4);
		int mm=0;for(int k=0;k<clustnectN_1S.rows;++k) mm+=clustnectN_1S.at<int>(k,0);
		if(mm==nn) break;
	}
	hzsort_Vert_1s();
	hzangle_Vert_1s();
}

void boxHZ::iterCluster_Hori(Mat &pwDistX,Mat &pwDistY,Mat &hzX,Mat &hzY,Mat &colorDist,Mat &nect,Mat &nectN, Mat &nectDire, double disThresh)
{
	int connect;
	for(int i=nectN.rows-1;i>0;--i)
	{/*std::cout<<"nect: "<<nect.row(i)<<std::endl;*/
		if(nectN.at<int>(i,0))
		{
			for(int j=i-1;j>=0;--j)
			{
				if(nectN.at<int>(j,0))
				{
					connect = 0;
					for(int ki=0;ki<nectN.at<int>(i,0);++ki)
						for(int kj=0;kj<nectN.at<int>(j,0);++kj)
						{
							double rx = double(hzX.at<int>(i,0))/double(hzX.at<int>(j,0));
							double ry = double(hzY.at<int>(i,0))/double(hzY.at<int>(j,0));
							double rr = rx*ry;
							if(colorDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<1500&&
								pwDistY.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))>disThresh&&
								pwDistY.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<4*min(hzY.at<int>(i,0),hzY.at<int>(j,0))&&
								pwDistX.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<-0.5*double(hzX.at<int>(i,0))&&
								rx>0.8&&rx<1.2&&ry>0.8&&ry<1.2)
							{
								connect = 1;
							}
						}
					if(connect)
					{
						for(int ki=0;ki<nectN.at<int>(i,0);++ki)
						{
							nect.at<int>(j,nectN.at<int>(j,0)++)=nect.at<int>(i,ki);
							nect.at<int>(i,ki)=0;
						}
						nectN.at<int>(i,0) = 0;
						nectDire.at<int>(j,0) = 1;
						nectDire.at<int>(i,0) = 0;
						/*std::cout<<"nect: "<<nectN<<std::endl<<std::endl;*/
						break;
					}
				}
			}
		}
	}
}

void boxHZ::iterCluster_Vert(Mat &pwDistX,Mat &pwDistY,Mat &hzX,Mat &hzY,Mat &colorDist,Mat &nect,Mat &nectN, Mat &nectDire, double disThresh)
{
	int connect;
	for(int i=nectN.rows-1;i>0;--i)
	{/*std::cout<<"nect: "<<nect.row(i)<<std::endl;*/
		if(nectN.at<int>(i,0) && (nectDire.at<int>(i,0)!=1))
		{
			for(int j=i-1;j>=0;--j)
			{
				if(nectN.at<int>(j,0) && (nectDire.at<int>(j,0)!=1))
				{
					connect = 0;
					for(int ki=0;ki<nectN.at<int>(i,0);++ki)
						for(int kj=0;kj<nectN.at<int>(j,0);++kj)
						{
							double rx = double(hzX.at<int>(i,0))/double(hzX.at<int>(j,0));
							double ry = double(hzY.at<int>(i,0))/double(hzY.at<int>(j,0));
							double rr = rx*ry;
							double cdis = colorDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj));
							double pwX = pwDistX.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj));
							double pwX1 = 4*min(hzX.at<int>(i,0),hzX.at<int>(j,0));
							double pwY = pwDistY.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj));
							double pwY1 = -0.5*double(hzY.at<int>(i,0));
							if(colorDist.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<1500&&
								pwDistX.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))>disThresh&&
								pwDistX.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<4*min(hzX.at<int>(i,0),hzX.at<int>(j,0))&&
								pwDistY.at<double>(nect.at<int>(i,ki),nect.at<int>(j,kj))<-0.5*double(hzY.at<int>(i,0))&&
								rx>0.8&&rx<1.2&&ry>0.8&&ry<1.2)//竖直
							{
								connect = 1;
							}
						}
						if(connect)
						{
							for(int ki=0;ki<nectN.at<int>(i,0);++ki)
							{
								nect.at<int>(j,nectN.at<int>(j,0)++)=nect.at<int>(i,ki);
								nect.at<int>(i,ki)=0;
							}
							nectN.at<int>(i,0)=0;
							nectDire.at<int>(j,0) = 2;
							nectDire.at<int>(i,0) = 0;
							/*std::cout<<"nect: "<<nectN<<std::endl<<std::endl;*/
							break;
						}
				}
			}
		}
	}
}


void boxHZ::selectCluster(Mat &dst, string dir, double minification, vector<vector<Point> >& POIWordMinP, vector<vector<Point> >& POIWordMaxP)
{
	Mat src = dst.clone();

	/* 加到画框中 */
	//string dirr;
	//dirr.assign(dir,0,dir.length()-name.length());
	//ofstream outfile;
	//outfile.open(dirr+"boxes.txt",ios::app);
	//if(!outfile.is_open()){cout<<"Can not open boxes.txt file!"<<endl;return;}
	//outfile<<name<<" ";

	int Num = clustnectN.rows;
	for(int i=0;i<Num;++i)
	{
		double edr = 0;
		if(clustnectN.at<int>(i,0)>1&&angle.at<double>(i,0)<0.5)
		{
			for(int k=0;k<clustnectN.at<int>(i,0);++k)
			{
				edr+=newedgeRatio[clustnect.at<int>(i,k)];
			}
			edr/=clustnectN.at<int>(i,0);
			if(edr>1.01)
			{
				int plantFlag = 0;
				int notPOIFlag = 0;
				int SVMFlag = 0;
				for(int k=0;k<clustnectN.at<int>(i,0);++k)
				{
					//int temp1 = clustnect.at<int>(i,k);
					//if ( newIsPlant[clustnect.at<int>(i,k)] == 1)
					//{
					//	plantFlag = 1;
					//	break;
					//}
					int boxHeight = newMaxP[clustnect.at<int>(i,k)].x+1 - (newMinP[clustnect.at<int>(i,k)].x-1) + 1;
					int boxWidth = newMaxP[clustnect.at<int>(i,k)].y+1 - (newMinP[clustnect.at<int>(i,k)].y-1) + 1;
					int centerY = (newMaxP[clustnect.at<int>(i,k)].x+1 + (newMinP[clustnect.at<int>(i,k)].x-1)) / 2;

					if (centerY > 1700)
					{
						notPOIFlag = 1;
						break;
					}
					if (centerY > 1400 && (boxHeight < 20 || boxHeight < 20))
					{
						notPOIFlag = 1;
						break;
					}
				}				
				if (/*plantFlag == 0 && */notPOIFlag == 0)
				{		
					vector<Point> onePOIChainMaxN;
					vector<Point> onePOIChainMinN;
					for(int k=0;k<clustnectN.at<int>(i,0);++k)
					{
						Point onePOIWordMax;
						onePOIWordMax.x = (int)((newMaxP[clustnect.at<int>(i,k)].x+1) / minification);
						onePOIWordMax.y = (int)((newMaxP[clustnect.at<int>(i,k)].y+1) / minification);
						onePOIChainMaxN.push_back(onePOIWordMax);
						Point onePOIWordMin;
						onePOIWordMin.x = (int)((newMinP[clustnect.at<int>(i,k)].x-1) / minification);
						onePOIWordMin.y = (int)((newMinP[clustnect.at<int>(i,k)].y-1) / minification);
						onePOIChainMinN.push_back(onePOIWordMin);						
					}
					POIWordMaxP.push_back(onePOIChainMaxN);
					POIWordMinP.push_back(onePOIChainMinN);
				}						
			}
		}		
	}
}


void boxHZ::selectCluster_1S(Mat &dst, string dir, double minification, vector<vector<Point> >& POIWordMinP, vector<vector<Point> >& POIWordMaxP)
{
	Mat src = dst.clone();

	/* 加到画框中 */
	//string dirr;
	//dirr.assign(dir,0,dir.length()-name.length());
	//ofstream outfile;
	//outfile.open(dirr+"boxes.txt",ios::app);
	//if(!outfile.is_open()){cout<<"Can not open boxes.txt file!"<<endl;return;}
	//outfile<<name<<" ";

	int Num = clustnectN_1S.rows;
	for(int i=0;i<Num;++i)
	{
		double edr = 0;
		if(clustnectN_1S.at<int>(i,0)>1&&angle_1S.at<double>(i,0)<0.5)
		{
			for(int k=0;k<clustnectN_1S.at<int>(i,0);++k)
			{
				edr+=edgeRatio[sharpWordSerials[clustnect_1S.at<int>(i,k)]];
			}
			edr/=clustnectN_1S.at<int>(i,0);
			if(edr>1.2)
			{
				int plantFlag = 0;
				int notPOIFlag = 0;
				int SVMFlag = 0;
				for(int k=0;k<clustnectN_1S.at<int>(i,0);++k)
				{
					//int temp1 = clustnect.at<int>(i,k);
					//if ( newIsPlant[clustnect.at<int>(i,k)] == 1)
					//{
					//	plantFlag = 1;
					//	break;
					//}
					int boxHeight = vMaxP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x+1 - (vMinP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x-1) + 1;
					int boxWidth = vMaxP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y+1 - (vMinP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y-1) + 1;
					int centerY = (vMaxP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x+1 + (vMinP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x-1)) / 2;

					if (centerY > 1700)
					{
						notPOIFlag = 1;
						break;
					}
					if (centerY > 1400 && (boxHeight < 20 || boxHeight < 20))
					{
						notPOIFlag = 1;
						break;
					}
				}				
				if (/*plantFlag == 0 && */notPOIFlag == 0)
				{		
					vector<Point> onePOIChainMaxN;
					vector<Point> onePOIChainMinN;
					for(int k=0;k<clustnectN_1S.at<int>(i,0);++k)
					{
						Point onePOIWordMax;
						onePOIWordMax.x = (int)((vMaxP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x+1) / minification);
						onePOIWordMax.y = (int)((vMaxP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y+1) / minification);
						onePOIChainMaxN.push_back(onePOIWordMax);
						Point onePOIWordMin;
						onePOIWordMin.x = (int)((vMinP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x-1) / minification);
						onePOIWordMin.y = (int)((vMinP[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y-1) / minification);
						onePOIChainMinN.push_back(onePOIWordMin);						
					}
					POIWordMaxP.push_back(onePOIChainMaxN);
					POIWordMinP.push_back(onePOIChainMinN);
				}						
			}
		}		
	}
}


void boxHZ::hzangle_Hori()/* 最好是用robustPCA计算投影方差，这里就简单计算一下 */
{
	int Num = clustnectN.rows;
	angle = Mat::zeros(Num,1,CV_64F);
	double mina,ag,dy,dx;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>1)
		{
			mina=0;
			for(int k=0;k<clustnectN.at<int>(i,0)-1;++k)
			{
				dy = abs(newPosition[clustnect.at<int>(i,k+1)].y-newPosition[clustnect.at<int>(i,k)].y);
				dx = abs(newPosition[clustnect.at<int>(i,k+1)].x-newPosition[clustnect.at<int>(i,k)].x);
				if(dy) ag = abs(dx/dy); else ag = 100000;
				mina = mina>ag?mina:ag;
			}
			angle.at<double>(i,0)=mina;
		}
	}
}

void boxHZ::hzangle_Vert()/* 最好是用robustPCA计算投影方差，这里就简单计算一下 */
{
	int Num = clustnectN.rows;
	//angle = Mat::zeros(Num,1,CV_64F);
	double mina,ag,dy,dx;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>1 && clustnectDirec.at<int>(i,0) == 2)
		{
			mina=0;
			for(int k=0;k<clustnectN.at<int>(i,0)-1;++k)
			{
				dy = abs(newPosition[clustnect.at<int>(i,k+1)].y-newPosition[clustnect.at<int>(i,k)].y);
				dx = abs(newPosition[clustnect.at<int>(i,k+1)].x-newPosition[clustnect.at<int>(i,k)].x);
				if(dx) ag = abs(dy/dx); else ag = 100000;
				mina = mina>ag?mina:ag;
			}
			angle.at<double>(i,0)=mina;
		}
	}
}

void boxHZ::hzangle_Hori_1s()/* 最好是用robustPCA计算投影方差，这里就简单计算一下 */
{
	int Num = clustnectN_1S.rows;
	angle_1S = Mat::zeros(Num,1,CV_64F);
	double mina,ag,dy,dx;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN_1S.at<int>(i,0)>1)
		{
			mina=0;
			for(int k=0;k<clustnectN_1S.at<int>(i,0)-1;++k)
			{
				dy = abs(Position[sharpWordSerials[clustnect_1S.at<int>(i,k+1)]].y-Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y);
				dx = abs(Position[sharpWordSerials[clustnect_1S.at<int>(i,k+1)]].x-Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x);
				if(dy) ag = abs(dx/dy); else ag = 100000;
				mina = mina>ag?mina:ag;
			}
			angle_1S.at<double>(i,0)=mina;
		}
	}
}

void boxHZ::hzangle_Vert_1s()/* 最好是用robustPCA计算投影方差，这里就简单计算一下 */
{
	int Num = clustnectN_1S.rows;
	//angle = Mat::zeros(Num,1,CV_64F);
	double mina,ag,dy,dx;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN_1S.at<int>(i,0)>1 && clustnectDirec_1S.at<int>(i,0) == 2)
		{
			mina=0;
			for(int k=0;k<clustnectN_1S.at<int>(i,0)-1;++k)
			{
				dy = abs(Position[sharpWordSerials[clustnect_1S.at<int>(i,k+1)]].y-Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y);
				dx = abs(Position[sharpWordSerials[clustnect_1S.at<int>(i,k+1)]].x-Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x);
				if(dx) ag = abs(dy/dx); else ag = 100000;
				mina = mina>ag?mina:ag;
			}
			angle_1S.at<double>(i,0)=mina;
		}
	}
}

void boxHZ::hzmeanW()
{
	int Num = clustnectN.rows;
	meanW = Mat::zeros(Num,1,CV_64F);
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>0)
		{
			for(int k=0;k<clustnectN.at<int>(i,0);++k)
				meanW.at<double>(i,0)+=newMaxP[clustnect.at<int>(i,k)].y-newMinP[clustnect.at<int>(i,k)].y;
			meanW.at<double>(i,0)/=clustnectN.at<int>(i,0);
		}
	}
}


struct ss{ int k; int y;};

bool sscomp(const ss& a,const ss& b){ return a.y<b.y;}

void boxHZ::hzsort_Hori()
{
	int Num = clustnectN.rows;
	ss y;
	vector<ss> pY;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>1)
		{
			for(int k=0;k<clustnectN.at<int>(i,0);++k)
			{
				y.k = clustnect.at<int>(i,k);
				y.y = newPosition[clustnect.at<int>(i,k)].y;
				pY.push_back(y);
			}
			sort(pY.begin(),pY.end(),sscomp);
			for(int k=clustnectN.at<int>(i,0);k>0;--k)
			{
				clustnect.at<int>(i,k-1) = pY[k-1].k;pY.pop_back();
				//std::cout<<newPosition[clustnect.at<int>(i,k-1)].y<<std::endl;
			}
			
		}
	}
}

void boxHZ::hzsort_Vert()
{
	int Num = clustnectN.rows;
	ss x;
	vector<ss> pX;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>1 && clustnectDirec.at<int>(i,0)== 2)
		{
			for(int k=0;k<clustnectN.at<int>(i,0);++k)
			{
				x.k = clustnect.at<int>(i,k);
				x.y = newPosition[clustnect.at<int>(i,k)].x;
				pX.push_back(x);
			}
			sort(pX.begin(),pX.end(),sscomp);
			for(int k=clustnectN.at<int>(i,0);k>0;--k)
			{
				clustnect.at<int>(i,k-1) = pX[k-1].k;pX.pop_back();
				//std::cout<<newPosition[clustnect.at<int>(i,k-1)].y<<std::endl;
			}

		}
	}
}

void boxHZ::hzsort_Hori_1s()
{
	int Num = clustnectN_1S.rows;
	ss y;
	vector<ss> pY;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN_1S.at<int>(i,0)>1)
		{
			for(int k=0;k<clustnectN_1S.at<int>(i,0);++k)
			{
				y.k = clustnect_1S.at<int>(i,k);
				y.y = Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].y;
				pY.push_back(y);
			}
			sort(pY.begin(),pY.end(),sscomp);
			for(int k=clustnectN_1S.at<int>(i,0);k>0;--k)
			{
				clustnect_1S.at<int>(i,k-1) = pY[k-1].k;pY.pop_back();
				//std::cout<<newPosition[clustnect.at<int>(i,k-1)].y<<std::endl;
			}

		}
	}
}

void boxHZ::hzsort_Vert_1s()
{
	int Num = clustnectN_1S.rows;
	ss x;
	vector<ss> pX;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN_1S.at<int>(i,0)>1 && clustnectDirec_1S.at<int>(i,0)== 2)
		{
			for(int k=0;k<clustnectN_1S.at<int>(i,0);++k)
			{
				x.k = clustnect_1S.at<int>(i,k);
				x.y = Position[sharpWordSerials[clustnect_1S.at<int>(i,k)]].x;
				pX.push_back(x);
			}
			sort(pX.begin(),pX.end(),sscomp);
			for(int k=clustnectN_1S.at<int>(i,0);k>0;--k)
			{
				clustnect_1S.at<int>(i,k-1) = pX[k-1].k;pX.pop_back();
				//std::cout<<newPosition[clustnect.at<int>(i,k-1)].y<<std::endl;
			}

		}
	}
}

void boxHZ::hzdist()
{/* 计算两个字之间的距离 */
	int Num = clustnectN.rows;
	for(int i=0;i<Num;++i)
	{
		if(clustnectN.at<int>(i,0)>2)
		{
			for(int k=0;k<clustnectN.at<int>(i,0);++k)
			{
				double mind=10000;
				for(int h=0;h<clustnectN.at<int>(i,0);++h)
				{
					if(h!=k)
					{
						
					}
				}
			}
		}
	}
}
