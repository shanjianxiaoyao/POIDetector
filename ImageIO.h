/*
    ImageIO.h
    TestRCA
  
    Created by Autonavi on 14-5-20.
    Copyright (c) 2014年 Autonavi. All rights reserved.
*/

#ifndef __TestRCA__ImageIO__
#define __TestRCA__ImageIO__
#include <algorithm>
#include <iostream>
#include "dirent.h"
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

/* - 从一个文件夹下读入所有图片文件，图片文件名存入vector：images_names */
void open_imgs_dir(const char *dir_name, std::vector<std::string> &images_names);

void getFiles( string path, vector<string>& files );
#endif /* defined(__TestRCA__ImageIO__) */
