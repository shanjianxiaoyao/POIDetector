#include "objdetect.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include "ImageIO.h"

#include "imageHZ.h"
using namespace cv;
using namespace std;

extern "C" {

/* 加载模型，生成文字和对应序号的映射表 */
/* mapFileName为映射表文件的文件 */
/* modelsDir为模型所在目录（该目录下包含所有的模型和一个TXT，txt中是部分加载的500个模型的序号） */
/* partLoadFlag为模型部分加载标志,默认值为0，全部加载；内存不足则设置为1，只加载较常用的500个 */
SVMDetector* initial(char * mapFileName_c, char * modelsDir_c)
{
	string mapFileName(mapFileName_c);
	string modelsDir(modelsDir_c);
	SVMDetector *svmdetect = NULL;
	svmdetect = new SVMDetector(mapFileName, modelsDir);
	return svmdetect;
}

POIDetecter* initial2(char *wordmap_c, char *configfile_c)
{
	/* 模型初始化 */
	string wordmap(wordmap_c);
	string configfile(configfile_c);
	POIDetecter *poidetecter = new POIDetecter(configfile, wordmap);
	return poidetecter;
}

char* detectimage(SVMDetector* svmdetect, POIDetecter* detecter, char* path, char* poiname)
{
	setNumThreads(1);
	string imgDir = string(path);
	string imgName = string(poiname);
	vector<OutBlock> objects/* = svmdetect->svmDetectPOI(imgDir, imgName)*/;

	vector<OutBlock> results = detecter->detectPoi(imgDir, imgName);
	/* 整合两部分的结果 */
	map<int, vector<OutBlock> > resultMap = detecter->mergeResult( objects, results);
	//map<int, vector<OutBlock> > resultMap = detecter->mergeResult(results);
	/* 整合最终的结果 */
    vector<OutBlock> res = detecter->judgeResult(resultMap);
    
	ostringstream detectorRes;
	detectorRes << poiname << " " << res.size();
	for( size_t i = 0; i < res.size(); i++ )
	{
		Rect object_rect = res[i].rect;
		int label = res[i].label;
		detectorRes << " " << label << " " << object_rect.x << " " << object_rect.y << " " << object_rect.width
			<< " "<< object_rect.height << " " << res[i].hanzi;
	}
    /*if(false)
    {
        Mat img = imread( path );
        Mat draw_img = img.clone();
        char tmp_buf[1000];
        for(size_t i=0; i< res.size(); i++)
        {
            Rect object_rect = res[i].rect;
            int label = res[i].label;
            sprintf(tmp_buf, "%d", label);
            rectangle( draw_img,  Point(object_rect.x, object_rect.y), Point(object_rect.x+object_rect.width, object_rect.y+object_rect.height), Scalar(0,255,0), 1 );
            putText( draw_img, string(tmp_buf), Point(object_rect.x, object_rect.y+object_rect.height), CV_FONT_HERSHEY_PLAIN, 1.5, Scalar(0,255,0), 2);
        }
        imwrite("./pic_draw/" + imgName, draw_img);
    }*/


	char result[2000];
	strcpy(result, detectorRes.str().c_str());
    //cout<< "dan:"<< result << endl;
	return result;
}

void releaseDetector(SVMDetector* svmdetect, POIDetecter* detecter)
{
	if (svmdetect!=NULL)
	{
		delete svmdetect;
		svmdetect = NULL;
	}
	if(detecter != NULL)
	{
		delete detecter;
		detecter = NULL;
	}
}

}

int main()
{ 
	SVMDetector* svmDetector = initial("", "");
	POIDetecter* cascadeDetector = initial2("D:\\code\\POI_cascadeAndSVM\\word_mapping.conf", "D:\\code\\POI_cascadeAndSVM\\cascade.model\\config.txt");

	detectimage(svmDetector, cascadeDetector, "D:\\pic\\ori\\B000A0BDBC,稻香村(刘家窑店).jpg", "B000A0BDBC,稻香村(刘家窑店).jpg" );

	releaseDetector(svmDetector, cascadeDetector);

	return 1;
}
