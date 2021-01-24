//---------------------------------【头文件空间声明部分】---------------------------------------
#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include "opencv2/imgproc/imgproc_c.h"
//#include "omp.h"
//-----------------------------------【命名空间声明部分】---------------------------------------
using namespace cv;
using namespace std;

#pragma region 全局变量和函数声明部分
//-----------------------------------【全局函变量声明部分】--------------------------------------
Point2f srcTriangle[3];
Point2f dstTriangle[3];//定义两组点，代表两个三角形，在仿射变换中用到
Mat warpMat(2, 3, CV_32FC1);//仿射变换用到的矩阵

//-----------------------------------【全局函数声明部分】--------------------------------------
static void ShowText();
static Mat strelDisk(int Radius);
static Mat thinImage(const Mat & src, const int maxIterations);
//static Mat remove_block(double minarea, Mat& src);
static void remove_block(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode);
static int xmin(Mat& inputImage);
static int xmax(Mat& inputImage);
static IplImage* Mat_to_IplImage(const Mat Img);
static Mat IplImage_to_Mat(const IplImage* pBinary);
static void Niblack(const IplImage *srcImg, IplImage *binImg, double k,int t);
static Mat NiBlack(const Mat Img, int t, double k, int l);
static void chao_thinimage(Mat &srcimage);
static Mat xuanzhuan(const Mat src, double angle, int a, int b);
static int xxmin(Mat& inputImage);
static int xxmax(Mat& inputImage);
static int yymin(Mat& inputImage);
static int yymax(Mat& inputImage);
static Mat claheGO(Mat src, int _step);

#pragma endregion


//-----------------------------------【main( )函数】------------------------------------------
int main(int argc, char** argv)
{
#pragma region 基础设置	

	double time0 = static_cast<double>(getTickCount());//记录起始时间
	system("color 2F");//改变console字体颜色	
	ShowText();//显示帮助文字

#pragma endregion

	/******************************* 图像预处理阶段 *******************************/

#pragma region 读入图像

	cout << "******* 图像预处理阶段开始 *******" << endl;

	//读入一张图片，载入图像
	Mat picture = imread("..\\1.bmp");
	if (!picture.data) { printf("读取图片错误！\n"); return false; }
	//显示载入的图片
	namedWindow("【原始图】");
	imshow("【原始图】", picture);
	cout<<"图像载入完成！"<<endl;

#pragma endregion

#pragma region 逆时针90度旋转图像

	//旋转图像
	Mat picture0, t;
	transpose(picture, t);
	flip(t, picture0, 0); //逆时针90°旋转图像
	namedWindow("【0图像旋转】");
	imshow("【0图像旋转】", picture0);//显示载入的图片
	cout << "图像旋转完成！" << endl;

#pragma endregion

#pragma region 转化为灰度图	

	Mat picture1;//参数定义	
	cvtColor(picture0, picture1, COLOR_BGR2GRAY);//将原图像转换为灰度图像	
	namedWindow("1图像灰度化");
	imshow("1图像灰度化", picture1);//显示载入的图片
	cout<<"图像灰度化完成！"<<endl;

#pragma endregion

#pragma region 图像大小和灰度归一化

	//均值滤波
	Mat picture2;
	blur(picture1, picture2, Size(3,3));
	namedWindow("2图像均值滤波");
	imshow("2图像均值滤波", picture2);
	cout<<"图像均值滤波完成！"<<endl;

	imwrite("..\\biye\\1均值滤波.bmp", picture2);

	//Sobel算子边缘检测
	Mat picture3;
	Sobel(picture2, picture3, -1, 2, 0, 3, 1, 0, BORDER_DEFAULT);//使用水平X方向Sobel
	namedWindow("3图像Sobel算子边缘检测");
	imshow("3图像Sobel算子边缘检测", picture3);
	cout << "图像Sobel算子边缘检测完成！" << endl;
	//matlab返回边缘检测阈值参数tv移植到opencv的问题我没有解决
	//边缘图存在噪声，通过判断连通区域面积的大小或者纵向延伸度将其去处
	imwrite("..\\biye\\2边缘检测.bmp", picture3);

	Mat picture4, picture5, picture6 ,picture7;
	Mat core1 = strelDisk(2);//构造形态学运算内核半径为2的圆
	morphologyEx(picture3, picture4, MORPH_CLOSE, core1);//实现闭运算
	threshold(picture4, picture5, 20, 255, THRESH_BINARY);//进行二值化处理，其中20为阈值，可以按需要调整
	//Mat picture6 = thinImage(picture5,-1);//对输入图像进行骨骼化、细化，迭代次数不限
	//Mat picture7 = remove_block(100.0, picture6);//去除二值图像的噪声
	//Mat picture8 = thinImage(picture7, -1);//对输入图像进行骨骼化、细化，迭代次数不限
	//Mat picture9 = remove_block(100.0, picture8);//去除二值图像的噪声（ps：此处并没有用，但是骨骼化需要此句，不可删除）
	chao_thinimage(picture5);
	picture6 = picture5;//对输入图像进行骨骼化、细化，迭代次数不限
	//imshow("6", picture6);
	remove_block(picture6, picture7, 100, 1, 1);//去除二值图像的噪声
	chao_thinimage(picture7);
	Mat picture8 = picture7;//对输入图像进行骨骼化、细化，迭代次数不限
	namedWindow("4图像去噪声");
	imshow("4图像噪声", picture4);
	imshow("4图去噪声", picture5);
	imshow("4像去噪声", picture6);
	imshow("4图像去噪声", picture8);
	cout << "图像去噪声完成！" << endl;
	imwrite("..\\biye\\3骨骼化.bmp", picture8);


	//寻找左区域手指边界横坐标最大值
	int x_min = xmin(picture8);
	cout << "手指左边界的横坐标值：" << x_min << endl;
	//寻找右区域手指边界横坐标最小值
	int x_max = xmax(picture8);
	cout << "手指右边界的横坐标值：" << x_max << endl;
	//提取ROI图像区域
	Mat imageROI = picture1(Range(40, picture1.rows-10),Range(x_min+6, x_max-6));//长宽我都进行了改变
	//Mat imageROI = picture1(Rect(x_min, 0, x_max - x_min, picture1.rows));//效果一样
	cout << "提取ROI图像区域完成！" << endl;
	namedWindow("5提取出的ROI图像");
	imshow("5提取出的ROI图像", imageROI);

	imwrite("..\\biye\\4ROI.bmp", imageROI);

	//进行尺寸归一化
	Mat picture10;
	int chang = imageROI.rows;
	int kuan = imageROI.cols;
	cout << "ROI区域未归一化尺寸前长为" << chang << endl;
	cout << "ROI区域未归一化尺寸前宽为" << kuan << endl;

	//设置源图像和目标图像上的三组点以计算仿射变换
	srcTriangle[0] = Point2f(0, 0);
	srcTriangle[1] = Point2f(static_cast<float>(imageROI.cols ), 0);
	srcTriangle[2] = Point2f(0, static_cast<float>(imageROI.rows ));
	dstTriangle[0] = Point2f(static_cast<float>(imageROI.cols*0.0), static_cast<float>(imageROI.rows*0.0));
	dstTriangle[1] = Point2f(static_cast<float>(imageROI.cols*180.0 / kuan), static_cast<float>(imageROI.rows*0.0));
	dstTriangle[2] = Point2f(static_cast<float>(imageROI.cols*0.0), static_cast<float>(imageROI.rows * 250.0 / chang));
	warpMat = getAffineTransform(srcTriangle, dstTriangle);//求得仿射变换矩阵	
	warpAffine(imageROI, picture10, warpMat, Size(180, 250), INTER_LINEAR);//对源图像应用仿射变换，采用双线性插值，如果要修改归一化图像大小一定要修改上面选点的坐标变换值
	
	//srcTriangle[0] = Point2f(0, 0);
	//srcTriangle[1] = Point2f(static_cast<float>(imageROI.cols), 0);
	//srcTriangle[2] = Point2f(0, static_cast<float>(imageROI.rows));
	//dstTriangle[0] = Point2f(static_cast<float>(imageROI.cols*0.0), static_cast<float>(imageROI.rows*0.0));
	//dstTriangle[1] = Point2f(static_cast<float>(imageROI.cols*80.0 / kuan), static_cast<float>(imageROI.rows*0.0));
	//dstTriangle[2] = Point2f(static_cast<float>(imageROI.cols*0.0), static_cast<float>(imageROI.rows * 310 / chang));
	//warpMat = getAffineTransform(srcTriangle, dstTriangle);//求得仿射变换矩阵	
	//warpAffine(imageROI, picture10, warpMat, Size(80, 310), INTER_LINEAR);//对源图像应用仿射变换，采用双线性插值，如果要修改归一化图像大小一定要修改上面选点的坐标变换值
	////以上有两种大小 60*210 & 80*310 图片可以选择，后又更改了

	cout << "ROI图像尺寸归一化完成！" << endl;
	cout << "ROI区域尺寸归一化后长：" << picture10.rows << endl;
	cout << "ROI区域尺寸归一化后宽：" << picture10.cols << endl;

	imwrite("..\\biye\\5大小归一化.bmp", picture10);

	//进行灰度归一化
	Mat picture11;
	equalizeHist(picture10, picture11);//直方图均衡化,即灰度归一化
	cout << "ROI图像灰度归一化完成！" << endl;
	namedWindow("6图像进行尺寸灰度归一化");
	imshow("6图像进行尺寸灰度归一化", picture11);

	cout << "******* 图像预处理阶段完成 *******" << endl;

#pragma endregion

	/******************************* 图像增强分割阶段 *******************************/

#pragma region 进行均值滤波并增强图像

	/*	本阶段主要采用基于模糊的分割图像算法，采用NiBlack算法对静脉特征进行提取*/
	cout << "******* 图像增强分割阶段开始 *******" << endl;

	//Mat picture11 = imread("..\\4.bmp");此处用于直接提取静脉ROI区域的图片读取用，仅测试代码用
	Mat picture12;
	//cvtColor(picture11, picture12, COLOR_BGR2GRAY);//将原图像转换为灰度图像	//如果是直接读取ROI区域图片需要先灰度化
	blur(picture11, picture12, Size(3, 3));//进行均值滤波操作
	cout << "图像均值滤波完成！" << endl;
	equalizeHist(picture12, picture12);//直方图均衡化
	imwrite("..\\biye\\6直方图均衡化.bmp", picture12);
	//picture12 = claheGO(picture11, 5);//CLAHE算法
	//imwrite("..\\biye\\7clahe直方图均衡化.bmp", picture12);
	cout << "图像增强完成！" << endl;
	imshow("6图像均值滤波", picture12);
	
#pragma endregion

#pragma region 图像分割

	//IplImage* picture13_1=Mat_to_IplImage(picture10);
	IplImage* picture13_1 = Mat_to_IplImage(picture12);//将Mat类转化为IplImage类
	IplImage* picture13_2 = cvCloneImage(picture13_1);//复制相同大小的IplImage类
	Niblack(picture13_1, picture13_2, -0.925 , 5 );//进行NiBlack算法处理图像，可调节参数，修正系数以及选择窗口，数据的调节设置对识别影响大，这里要反复调节后两个参数，建议写后面回调函数调整参数观察最优值
	Mat picture13 = IplImage_to_Mat(picture13_2);////将IplImage类转化为Mat类
	imshow("7图像进行NiBlack算法处理", picture13);
	cout << "图像NiBlack算法处理完成！" << endl;

	imwrite("..\\biye\\8niblack.bmp", picture13);

	Mat picture14;
	remove_block(picture13, picture14, 15, 1, 1);//去除二值图像的噪声，去除面积值可调，影响大
	//Mat picture14 = remove_block(10.0, picture13);//去除二值图像的噪声，去除面积值可调，影响大
	//morphologyEx(picture14, picture14, MORPH_CLOSE, MORPH_ELLIPSE);//实现闭运算
	imshow("8图像分割处理", picture14);
	cout << "图像分割处理完成！" << endl;
	imwrite("..\\biye\\9去除噪声.bmp", picture14);

	//Mat M = NiBlack(picture12, 2, 50, INT_MAX);//貌似写的Mat类的的niblank算法不太好用，选择放弃
	//imshow("窗口", M);

	cout << "******* 图像增强分割阶段完成 *******" << endl;

#pragma endregion

	/******************************* 图像后处理阶段 *******************************/

#pragma region 图像滤波去噪

	cout << "******* 图像骨骼化处理开始 *******" << endl;


	Mat picture15, picture16;
	medianBlur(picture14, picture15, 5);//中值滤波

	remove_block(picture15, picture15, 40, 1, 1);//去除噪声

	//Mat picture15, picture16;
	//medianBlur(picture14, picture15, 5);//中值滤波
	//picture15 = remove_block(22.0, picture15);//去除噪声
	//morphologyEx(picture15, picture16, MORPH_CLOSE, NULL);//实现闭运算
	imshow("9图像滤波去噪处理", picture15);
	cout << "图像滤波去噪处理完成！" << endl;

	imwrite("..\\biye\\10滤波去除噪声.bmp", picture15);


#pragma endregion

#pragma region 图像骨骼化

	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(8, 25));//构建椭圆算子，参数可调，影响大
	dilate(picture15, picture16, element);//进行形态学膨胀
	imwrite("..\\biye\\11形态学膨胀.bmp", picture16);

	Mat picture17;
	medianBlur(picture16, picture17, 7);//中值滤波
	remove_block(picture17, picture17, 200, 1, 1);//去除大颗粒噪声，这不实在,可注释
	//picture17 = remove_block(120, picture17);//去除大颗粒噪声，这不实在,可注释
	imwrite("..\\biye\\12去除大颗粒.bmp", picture17);
	//Canny(picture17, picture18, 3, 9, 3);
	//morphologyEx(picture17, picture18, MORPH_ERODE, Size(6, 6));//实现闭运算
//	Mat picture19, picture20;
//	Mat picture19 = thinImage(picture17, -1);//对输入图像进行骨骼化、细化，迭代次数不限
	imshow("10图像骨骼化处理1", picture15);
	imshow("10图像骨骼化处理2", picture16);
	imshow("10图像骨骼化处理3", picture17);
	chao_thinimage(picture17);
	//blur(picture17, picture18,Size(3,3));
	imshow("10图像骨骼化处理4", picture17);
	//imshow("图像骨骼化处理", picture18);
	cout << "图像骨骼化处理完成！" << endl;
	imwrite("..\\tu_data\\骨骼化图片.bmp", picture17);
	imwrite("..\\biye\\13骨骼化.bmp", picture17);

	cout << "图像保存完成！" << endl;
	
	cout << "******* 图像骨骼化处理完成 *******" << endl;

#pragma endregion


time0 = ((double)getTickCount() - time0) / getTickFrequency();//计算运行时间并输出
cout << "\n总运行时间为： " << time0 << "秒" << endl << endl;  //输出运行时间

	//等待任意按键按下
	waitKey(0);
	return 0;
}
//-----------------------------------【调用构造的全局子函数】------------------------------------------
#pragma region 输出信息函数ShowText( )

//-----------------------------------【ShowHelpText( )函数】----------------------------------  
static void ShowText()//输出一些帮助信息  
{
	//输出基本信息和OpenCV版本
	printf("\n\t\t\t	OpenCV指静脉骨架提取项目代码1.0版本  \n");
	printf("\n\t\t\t	2018年3月  CIDP_425_Lab	 \n"); 
	printf("\n\t\t\t	By@JXH   github:j1o2h3n\n");
	printf("\n\t\t\t	开发所用IDE版本：Visual Studio 2013	 \n");
	printf("\n\t\t\t	当前使用的OpenCV版本为：" CV_VERSION);
	printf("\n\n\n  -----------------------------------------------------------------------------------------------------------\n\n\n");
}

#pragma endregion

#pragma region 构造形态学运算内核半径为R的圆函数strelDisk()

//实现matlab中strel('disk',Radius)函数 
static Mat strelDisk(int Radius)
{
	int borderWidth; Mat sel; int m, n;
	switch (Radius){
	case 1:
	case 2:
		if (Radius == 1)
			borderWidth = 1;
		else
			borderWidth = 2;
		sel = Mat((2 * Radius + 1), (2 * Radius + 1), CV_8U, cv::Scalar(1));
		break;//当半径为1时是3X3的 ,当半径为2时是5X5的  
	case 3:
		borderWidth = 0;
		sel = Mat((2 * Radius - 1), (2 * Radius - 1), CV_8U, cv::Scalar(1));
		break;
	default:
		n = Radius / 7; m = Radius % 7;
		if (m == 0 || m >= 4)
			borderWidth = 2 * (2 * n + 1);
		else
			borderWidth = 2 * 2 * n;
		sel = Mat((2 * Radius - 1), (2 * Radius - 1), CV_8U, cv::Scalar(1));
		break;
	}
	for (int i = 0; i < borderWidth; i++){
		for (int j = 0; j < borderWidth; j++){
			if (i + j < borderWidth){
				sel.at<uchar>(i, j) = 0;
				sel.at<uchar>(i, sel.cols - 1 - j) = 0;
				sel.at<uchar>(sel.rows - 1 - i, j) = 0;
				sel.at<uchar>(sel.rows - 1 - i, sel.cols - 1 - j) = 0;
			}
		}
	}
	return sel;
}

#pragma endregion

#pragma region 对输入图像进行细化骨骼化的函数thinImage()

/**
* @brief 对输入图像进行细化、骨骼化
* @param src为输入图像,用cvThreshold函数处理过的8位灰度图像格式，元素中只有0与1,1代表有元素，0代表为空白
* @param maxIterations限制迭代次数，如果不进行限制，默认为-1，代表不限制迭代次数，直到获得最终结果
* @return 为对src细化后的输出图像,格式与src格式相同，元素中只有0与1,1代表有元素，0代表为空白
*/
static Mat thinImage(const Mat &src, const int maxIterations = -1)
{
	assert(src.type() == CV_8UC1);
	Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	int count = 0;  //记录迭代次数    
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //限制次数并且迭代次数到达    
			break;
		std::vector<uchar *> mFlag; //用于标记需要删除的点    
		//对点标记    
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记    
				//  p9 p2 p3    
				//  p8 p1 p4    
				//  p7 p6 p5    
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//标记    
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除    
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}

		//对点标记    
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//如果满足四个条件，进行标记    
				//  p9 p2 p3    
				//  p8 p1 p4    
				//  p7 p6 p5    
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//标记    
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//将标记的点删除    
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//直到没有点满足，算法结束    
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//将mFlag清空    
		}
	}
	return dst;
}

#pragma endregion

//#pragma region 删除二值化图像中面积较小的区域remove_block()
//
//static Mat remove_block(double minarea, Mat& src)
//{
//	CvSeq* contour = NULL;
//	double tmparea = 0.0;
//	CvMemStorage* storage = cvCreateMemStorage(0);
//	//const char* tmpaddsum1 = tmp_string.c_str();  
//
//	//IplImage* img_src = cvLoadImage(tmpaddsum1, CV_LOAD_IMAGE_ANYCOLOR);  
//	IplImage* img_src = &IplImage(src);
//	IplImage* img_Clone = cvCloneImage(img_src);
//	//访问二值图像每个点的值  
//	uchar *pp;
//	IplImage* img_dst = cvCreateImage(cvGetSize(img_src), IPL_DEPTH_8U, 1);
//
//	//-----搜索2值图中的轮廓，并从轮廓树中删除面积小于minarea的区域-----//  
//	//CvScalar color = cvScalar(255, 0, 0);//CV_RGB(128,0,0)  
//	CvContourScanner scanner = NULL;
//	scanner = cvStartFindContours(img_src, storage, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE, cvPoint(0, 0));
//	//开始遍历轮廓树  
//	CvRect rect;
//
//	while (contour = cvFindNextContour(scanner)){
//		tmparea = fabs(cvContourArea(contour));
//		rect = cvBoundingRect(contour, 0);
//
//		if (tmparea < minarea){
//			//当连通区域的中心点为白色时，而且面积较小则用黑色进行填充  
//			pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*(rect.y + rect.height / 2) + rect.x + rect.width / 2);
//			if (pp[0] == 255){
//				for (int y = rect.y; y<rect.y + rect.height; y++)
//				{
//					for (int x = rect.x; x<rect.x + rect.width; x++)
//					{
//						pp = (uchar*)(img_Clone->imageData + img_Clone->widthStep*y + x);
//
//						if (pp[0] == 255)
//						{
//							pp[0] = 0;
//						}
//					}
//				}
//			}
//
//		}
//	}
//	Mat dst_img = cvarrToMat(img_Clone);
//	if (dst_img.channels() == 3)
//		cvtColor(dst_img, dst_img, CV_RGB2GRAY);
//	return dst_img;
//}
//
//#pragma endregion

#pragma region 删除二值化图像中面积较小的区域remove_block()

static void remove_block(Mat &Src, Mat &Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;
	Dst = Src;
	//新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查,2代表检查不合格（需要反转颜色），3代表检查合格或不需检查   
	//初始化的图像全部为0，未检查  
	Mat PointLabel = Mat::zeros(Src.size(), CV_8UC1);
	if (CheckMode == 1)//去除小连通区域的白色点  
	{
		//cout << "去除小连通域.";
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) < 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//将背景黑色点标记为合格，像素为3  
				}
			}
		}
	}
	else//去除孔洞，黑色点像素  
	{
		//cout << "去除孔洞";
		for (int i = 0; i < Src.rows; i++)
		{
			for (int j = 0; j < Src.cols; j++)
			{
				if (Src.at<uchar>(i, j) > 10)
				{
					PointLabel.at<uchar>(i, j) = 3;//如果原图是白色区域，标记为合格，像素为3  
				}
			}
		}
	}


	vector<Point2i>NeihborPos;//将邻域压进容器  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));


	if (NeihborMode == 1)
	{
		//cout << "Neighbor mode: 8邻域." << endl;
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	else int a = 0;//cout << "Neighbor mode: 4邻域." << endl;
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//开始检测  
	for (int i = 0; i < Src.rows; i++)
	{
		for (int j = 0; j < Src.cols; j++)
		{
			if (PointLabel.at<uchar>(i, j) == 0)//标签图像像素点为0，表示还未检查的不合格点  
			{   //开始检查  
				vector<Point2i>GrowBuffer;//记录检查像素点的个数  
				GrowBuffer.push_back(Point2i(j, i));
				PointLabel.at<uchar>(i, j) = 1;//标记为正在检查  
				int CheckResult = 0;

				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					for (int q = 0; q < NeihborCount; q++)
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX < Src.cols&&CurrY >= 0 && CurrY < Src.rows)  //防止越界    
						{
							if (PointLabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //邻域点加入buffer    
								PointLabel.at<uchar>(CurrY, CurrX) = 1;           //更新邻域点的检查标签，避免重复检查    
							}
						}
					}
				}
				if (GrowBuffer.size() > AreaLimit) //判断结果（是否超出限定的大小），1为未超出，2为超出    
					CheckResult = 2;
				else
				{
					CheckResult = 1;
					RemoveCount++;//记录有多少区域被去除  
				}

				for (int z = 0; z < GrowBuffer.size(); z++)
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					PointLabel.at<uchar>(CurrY, CurrX) += CheckResult;//标记不合格的像素点，像素值为2  
				}
				//********结束该点处的检查**********    
			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//开始反转面积过小的区域    
	for (int i = 0; i < Src.rows; ++i)
	{
		for (int j = 0; j < Src.cols; ++j)
		{
			if (PointLabel.at<uchar>(i, j) == 2)
			{
				Dst.at<uchar>(i, j) = CheckMode;
			}
			else if (PointLabel.at<uchar>(i, j) == 3)
			{

				Dst.at<uchar>(i, j) = Src.at<uchar>(i, j);

			}
		}
	}


}

#pragma endregion

#pragma region 寻找左区域边界最大值纵坐标xmin()

static int xmin(Mat& inputImage)
{
	int x_min=0;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = 0; j < inputImage.cols / 2; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (x_min<j)
				{
					x_min = j;
				}
			}
		}
	}
	return x_min;
}


#pragma endregion

#pragma region 寻找右区域边界最小值纵坐标xmax()

static int xmax(Mat& inputImage)
{
	int x_max = inputImage.cols;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = inputImage.cols / 2; j < inputImage.cols; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (x_max>j)
				{
					x_max = j;
				}
			}
		}
	}
	return x_max;
}


#pragma endregion

#pragma region 采用IplImage类型处理图像Nilblack二值化算法函数Niblack()
/*************************************************************
Nilblack二值化算法是针对全局阈值发在图像和背景灰度相差明显时
容易忽略细节的问题进行优化，是一种常用的比较有效的局部阈值
算法。这种算法的基本思想是对预想中的每一个点，在它的R*R领域内，
计算领域里的像素点的均值和方差，然后下式计算阈值进行二值化：
T(x,y) = m(x,y) + k* s(x,y)其中对于像素点坐标(x,y),T(x,y) 为该
点阈值，m(x,y)为该点的R*R领域内像素点的像素均值，s(x,y)为该点
R*R领域内像素标准方差，k为修正系数(通常取-0.1),t是选择算法窗口大小。
*************************************************************/
static void Niblack(const IplImage *srcImg, IplImage *binImg, double k,int t)
{
	int m = srcImg->width;//宽
	int n = srcImg->height;//高
	int Niblack_q=t;//算子行
	int Niblack_s=t;//算子列
	

	for (int i = 0; i < n; i++)//行循环
	{
		uchar *srcptr = (uchar*)(srcImg->imageData + srcImg->widthStep * i);//相关行i最左边的像素的指针srcptr
		uchar *dstptr = (uchar*)(binImg->imageData + binImg->widthStep * i);
		for (int j = 0; j < m; j++)//列循环
		{
			int begin_y = i - Niblack_q, begin_x = j - Niblack_s;
			int end_y = i + Niblack_q, end_x = j + Niblack_s;
			if (begin_y < 0) begin_y = 0;
			if (begin_x < 0) begin_x = 0;
			if (end_y > n) end_y = n;
			if (end_x > m) end_x = m;

			double mean = 0;
			double std = 0;
			int total = (end_y - begin_y) * (end_x - begin_x);  //该领域内总的像素点数，此处应该为end-begin+1的值，但我懒得改了  

			for (int y = begin_y; y<= end_y; y++)//行
			{
				uchar *ptr = (uchar*)(srcImg->imageData + srcImg->widthStep * y);
				for (int x = begin_x; x <= end_x; x++)//列
				{
					mean += ptr[x];
				}
			}  //计算在该小领域内灰度值总和  

			mean = mean / total;  //该领域的平均灰度  

			for (int y = begin_y; y <= end_y; y++)
			{
				uchar *ptr = (uchar*)(srcImg->imageData + srcImg->widthStep * y);
				for (int x = begin_x; x <= end_x; x++)
				{
					double sq = (mean - ptr[x]) * (mean - ptr[x]);
					std += sq;
				}
			}   //求出该领域内总的方差和  

			std /= total;
			std = sqrt(std);  //求出标准差  

			double threshold = mean + k * std;  //求出所得到的的阈值  

			if (srcptr[j] > threshold-1)
				dstptr[j] = 0;
			else
				dstptr[j] = 255;
		}
	}
}

#pragma endregion

#pragma region 将Mat转化为IplImage类型的函数Mat_to_IplImage()

static IplImage* Mat_to_IplImage(const Mat Img)
{
	//Mat—>IplImage  
	//浅拷贝:  
	IplImage* pBinary = &IplImage(Img);
	//深拷贝只要再加一次复制数据  
	IplImage *input = cvCloneImage(pBinary);
	return input;
}

#pragma endregion

#pragma region 将IplImage转化为Mat类型的函数IplImage_to_Mat()

static Mat IplImage_to_Mat(const IplImage *pBinary)
{
	//IplImage—>Mat  
	//浅拷贝：  
	Mat Img;
	Img = cvarrToMat(pBinary);
	//深拷贝只需要再在Mat里创建一个新的Mat对象，然后进行数据的复制，再用上述的函数进行数据头的复制（浅拷贝）  
	Mat ImgTemp;
	ImgTemp = Img.clone();
	return ImgTemp;
}

#pragma endregion

#pragma region 采用Mat类型操作图像写的NiBlack二值化算法函数NiBlack()

//此方法我不想再使用，后有来者选择性借鉴，尾部代码我没有煞尾完
static  Mat NiBlack(const Mat Img, int t, double k, int l)
{
	int m = Img.rows;//行
	int n = Img.cols;//列
	//Mat M = Mat::zeros((Img.rows + 2 * t),( Img.cols + 2 * t),CV_8UC1);//构建零矩阵
	//Mat yanmo = M(Rect(t , t , Img.cols, Img.rows));
	//Img.copyTo(yanmo, Img);//图像掩膜
	Mat M;
	cv::copyMakeBorder(Img, M, t, t, t, t, cv::BORDER_CONSTANT, 0);//扩宽图像边界，比掩膜靠谱
	Mat b = Mat::zeros(Img.rows, Img.cols, CV_8UC1);
	Mat h = Mat::zeros(Img.rows, Img.cols, CV_8UC1);
	for (int i = t; i<(m + t); i++)//at<uchar>(0, 0)开始
	{
		for (int j = t ; j <(n + t); j++)
		{
			int begin_y = i - t, begin_x = j - t;
			int end_y = i +t, end_x = j + t;
			double mean = 0;
			double std = 0;
			int total = ((2*t)^2);  //该领域内总的像素点数,按照matlab而来 
			
			for (int y = begin_y; y<= end_y; y++)
			{
				for (int x = begin_x; x <= end_x; x++)
				{
					mean = mean + M.at<uchar>(y, x);
				}
			}  //计算在该小领域内灰度值总和  


			mean = mean / total;  //该领域的平均灰度  

			for (int y = begin_y; y <= end_y; y++)
			{
				for (int x = begin_x; x <= end_x; x++)
				{
					double sq = (M.at<uchar>(y, x) - mean) * (M.at<uchar>(y, x) - mean);
					std = std + sq;
				}
			}   //求出该领域内总的方差和  

			std = std / total;
			std = sqrt(std);  //求出标准差  

			double threshold = k * std +  mean;  //求出所得到的的阈值
			//M.at<uchar>(i-t, j-t) = threshold;
			M.at<uchar>(i , j ) = threshold;
			//M.at<uchar>(i, j) = mean;
		}
	}
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (Img.at<uchar>(i, j) <= M.at<uchar>(i, j))
			{
				/*h.at<uchar>(i, j) = 255;*/ h.at<uchar>(i, j) = Img.at<uchar>(i, j)*double((Img.at<uchar>(i, j) / M.at<uchar>(i, j)) ^ l);
			}
			else //if (M.at<uchar>(i, j) != 255)
			{
				/*h.at<uchar>(i, j) = 0;*/ h.at<uchar>(i, j) = 256 - ((255 - Img.at<uchar>(i, j))*double(((255 - Img.at<uchar>(i, j)) / (255 - M.at<uchar>(i, j))) ^ l));
			}
			//else if (M.at<uchar>(i, j) == 255)
			//{
				///*h.at<uchar>(i, j) = 0;*/ h.at<uchar>(i, j) = 0;
			//}
		}
	}

	//Mat shuchu;	
	return M;
}

#pragma endregion

#pragma region 骨骼化二代函数chao_thinimage()

//建议后面的人用这个把前面的那个替换掉，前面那个不是很好，而且还有小问题

static void chao_thinimage(Mat &srcimage)//单通道、二值化后的图像  
{
	vector<Point> deletelist1;
	int Zhangmude[9];
	int nl = srcimage.rows;
	int nc = srcimage.cols;
	while (true)
	{
		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[7] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[5] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();

		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[3] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[1] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();
	}
}

#pragma endregion

#pragma region 图像旋转函数xuanzhuan()
//第一个参数输入图像，第二个参数旋转角度，第三四参数是裁剪旋转图像后保留的长、宽,长宽应小于原长
static Mat xuanzhuan(const Mat src,double angle,int a,int b)
{
	Point2f center(src.cols / 2, src.rows / 2);
	Mat rot = getRotationMatrix2D(center, angle, 1);
	Rect bbox = RotatedRect(center, src.size(), angle).boundingRect();

	rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	Mat dst;
	//旋转
	warpAffine(src, dst, rot, bbox.size());
	//裁剪
	dst = dst(Range(((dst.rows - a) / 2), ((dst.rows + a) / 2)), Range(((dst.cols - b) / 2), ((dst.cols + b) / 2)));
	
	return dst;
}

#pragma endregion

#pragma region 区域边界最小值纵坐标xxmin()

static int xxmin(Mat& inputImage)
{
	int xx_min = inputImage.cols;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = 0; j < inputImage.cols; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (j<xx_min)
				{
					xx_min = j;
				}
			}
		}
	}
	return xx_min;
}


#pragma endregion

#pragma region 区域边界最大值纵坐标xxmax()

static int xxmax(Mat& inputImage)
{
	int xx_max = 0;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = 0; j < inputImage.cols; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (j>xx_max)
				{
					xx_max = j;
				}
			}
		}
	}
	return xx_max;
}


#pragma endregion

#pragma region 区域边界最小值纵坐标yymin()

static int yymin(Mat& inputImage)
{
	int yy_min = inputImage.rows;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = 0; j < inputImage.cols; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (i<yy_min)
				{
					yy_min = i;
				}
			}
		}
	}
	return yy_min;
}

#pragma endregion

#pragma region 区域边界最小值纵坐标yymax()

static int yymax(Mat& inputImage)
{
	int yy_max = 0;
	for (int i = 0; i < inputImage.rows; ++i)//行
	{
		for (int j = 0; j < inputImage.cols; ++j)//列
		{
			if (inputImage.at<uchar>(i, j))
			{
				if (i>yy_max)
				{
					yy_max = i;
				}
			}
		}
	}
	return yy_max;
}

#pragma endregion

#pragma region CLAHE算法claheGO()

static Mat claheGO(Mat src, int _step)//_step小于等于8
{
	//int _step = 5;//此参数可调
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //每个小格子的长和宽
	int height_block = height / block;
	//存储各个直方图  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//分块
	int total = width_block * height_block;
	for (int i = 0; i < block; i++)
	{
		for (int j = 0; j < block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//遍历小块,计算直方图
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//裁剪和增加操作，也就是clahe中的cl部分
			//这里的参数 对应《Gem》上面 fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			//关于参数如何选择，需要进行讨论。不同的结果进行讨论
			//关于全局的时候，这里的这个cl如何算，需要进行讨论 
			int LIMIT = 40 * average;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely  
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}
			//计算累积分布直方图  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//计算变换后的像素值  
	//根据像素点的位置，选择不同的计算方法  
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			//four coners  
			if (i <= width_block / 2 && j <= height_block / 2)
			{
				int num = 0;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i <= width_block / 2 && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * (block - 1);
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j <= height_block / 2) {
				int num = block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			//four edges except coners  
			else if (i <= width_block / 2)
			{
				//线性插值  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2)) {
				//线性插值  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1)*height_block + height_block / 2)) {
				//线性插值  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//双线性插值
			else {
				int num_i = (i - width_block / 2) / width_block;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				int num3 = num1 + block;
				int num4 = num2 + block;
				float u = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float v = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				CLAHE_GO.at<uchar>(j, i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j, i)] +
					(1 - v)*(1 - u)*C2[num1][CLAHE_GO.at<uchar>(j, i)] +
					u * (1 - v)*C2[num2][CLAHE_GO.at<uchar>(j, i)] +
					v * (1 - u)*C2[num3][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//最后这步，类似高斯平滑
			CLAHE_GO.at<uchar>(j, i) = CLAHE_GO.at<uchar>(j, i) + (CLAHE_GO.at<uchar>(j, i) << 8) + (CLAHE_GO.at<uchar>(j, i) << 16);
		}
	}
	return CLAHE_GO;
}

#pragma endregion