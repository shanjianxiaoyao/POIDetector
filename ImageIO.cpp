//
//  ImageIO.cpp
//  TestRCA
//
//  Created by Autonavi on 14-5-20.
//  Copyright (c) 2014年 Autonavi. All rights reserved.
//

#include "ImageIO.h"
bool hasEnding (std::string const &fullString, std::string const &ending)
{
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
	} else {
		return false;
	}
}

bool hasEndingLower (string const &fullString_, string const &_ending)
{
	string fullstring = fullString_, ending = _ending;
	transform(fullString_.begin(),fullString_.end(),fullstring.begin(),::tolower); // to lower
	return hasEnding(fullstring,ending);
}

void open_imgs_dir(const char *dir_name, std::vector<std::string> &images_names)
{
    if(dir_name == NULL)
        return;
    
    string dir_name_ = string(dir_name);
    vector<string> files_;
    
    DIR *dp;
    struct dirent *ep;
    dp = opendir(dir_name);
    if(dp != NULL)
    {
        while((ep = readdir(dp)))
        {
            if(ep->d_name[0] != '.')
                files_.push_back(ep->d_name);
            
        }
        closedir(dp);
    }
    else
    {
        cerr<<"can't open the dir: "<<dir_name_<<endl;
        return;
    }
    for(size_t i = 0; i < files_.size(); ++i)
    {
        if(files_[i][0] == '.' || !(hasEndingLower(files_[i],"jpg")||hasEndingLower(files_[i],"png")))
            continue;
		images_names.push_back(files_[i]);
    }
}

#ifdef WINVER 
#include<io.h>
void getFiles( string path, vector<string>& files )  
{  
	//文件句柄  
	long   hFile   =   0;  
	//文件信息  
	struct _finddata_t fileinfo;  
	string p;  
	if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
	{  
		do  
		{  
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if((fileinfo.attrib &  _A_SUBDIR))  
			{  
				if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
					getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
			}  
			else  
			{  
				//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
				files.push_back(fileinfo.name);  
			}  
		}while(_findnext(hFile, &fileinfo)  == 0);  
		_findclose(hFile);  
	}  
}  
#endif