#include "opencv2/opencv.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/highgui/highgui.hpp"
#include <cctype>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "dirent.h"
#include <unistd.h>
#include <string>
#include <ctime>
#include <cstdlib>
#include <chrono>
#include "ORBextractor.h"
#include "DBoW2/DBoW2/TemplatedVocabulary.h"
#include "DBoW2/DBoW2/FORB.h"
#include "DBoW2/DBoW2/BowVector.h"
#include "DBoW2/DBoW2/FeatureVector.h"

using namespace cv;
using namespace std;
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::system_clock::now();
    }

    double toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

  private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};

std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);
    for (int j=0;j<Descriptors.rows;j++)
        vDesc.push_back(Descriptors.row(j));

    return vDesc;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

vector<Point2f> forw_pts, n_pts;

void rejectWithF()
{
    if (forw_pts.size() >= 8)
    {
        vector<uchar> status;
        cv::findFundamentalMat(n_pts, forw_pts, cv::FM_RANSAC, 1.0, 0.99, status);
        int size_a = n_pts.size();
        reduceVector(n_pts, status);
        reduceVector(forw_pts, status);
    }
}

int main(int argc, char* argv[])
{

    std::cout << "OpenCV version: "
            << CV_MAJOR_VERSION << "." 
            << CV_MINOR_VERSION << "."
            << CV_SUBMINOR_VERSION
            << std::endl;

    std::string inputDirectory = "/home/peiliang/workspace/tracking_sim/image";
    DIR *directory = opendir (inputDirectory.c_str());
    struct dirent *_dirent = NULL;
    if(directory == NULL)
    {
        printf("Cannot open Input Folder\n");
        return 1;
    }
    ORBVocabulary* mpVocabulary;
    mpVocabulary = new ORBVocabulary();
    std::string strVocFile = "/home/peiliang/workspace/tracking_sim/DBoW2/ORBvoc.txt";
    TicToc t_vob;
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    printf("t vob = %lf\n",t_vob.toc());
    cout << "Vocabulary loaded!" << endl << endl;

    std::string fileName1 = inputDirectory + "/1.JPG";
    cv::Mat rawImage1 = cv::imread(fileName1.c_str());
    cv::Mat gray1;
    cv::cvtColor(rawImage1, gray1, CV_RGBA2GRAY);
    TicToc t_goodfeature;
    goodFeaturesToTrack(gray1, n_pts, 50, 0.10, 30, cv::noArray(), 3, false, 0.04);
    printf("t goodfeature %d : %lf ms\n", n_pts.size(), t_goodfeature.toc());
    for (auto &p : n_pts)
    {
        cv::circle(rawImage1, p, 3, cv::Scalar(0,0,255));
    }
    vector<Point2f> pre_pts = n_pts;

    std::string fileName2 = inputDirectory + "/5.JPG";
    cv::Mat rawImage2 = cv::imread(fileName2.c_str());
    cv::Mat gray2;
    cv::cvtColor(rawImage2, gray2, CV_RGBA2GRAY);

    std::string fileName3 = inputDirectory + "/3.JPG";
    cv::Mat rawImage3 = cv::imread(fileName3.c_str());

   
    vector<KeyPoint> n_kpts;
    Mat n_des;
    TicToc t_fast;
    //ORB_SLAM::ORBextractor orb1(200,1.2,8,1,20);
    ORB orb1(500,1.2,8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    cv::Mat indexMap(640,480,CV_32F);
    indexMap.setTo(0);
    orb1(gray1,cv::noArray(),n_kpts,n_des,false);
    int ferature_cnt = 0;
    for(auto &it : n_kpts)
    {
        indexMap.at<float>(it.pt.x, it.pt.y) = (float)ferature_cnt;
        ferature_cnt++;
        //printf("featrue: %d\n",ferature_cnt++);
    }
    cout << n_des.size() << endl;
    cout << n_kpts.size() << endl;
    namedWindow("index", CV_WINDOW_NORMAL);
    imshow("index",indexMap);
    printf("debug\n");
    
    //calculate bow of 1st frame
    TicToc t_bow1;
    vector<cv::Mat> vCurrentDesc = toDescriptorVector(n_des);
    DBoW2::BowVector BowVec1;
    DBoW2::FeatureVector FeatVec1;
    mpVocabulary->transform(vCurrentDesc,BowVec1,FeatVec1,4);
    printf("t bow1 %lf\n",t_bow1.toc());
    /*
    int orb_cnt = 0;
    for(int ix = 0; ix < 480; ix++)
            for(int iy = 0; iy < 640; iy++)
            {
                if(indexMap.at<float>(ix,iy)!=0)
                    cout << orb_cnt++ << ": " << indexMap.at<float>(ix,iy) << endl;
            }
    */
    vector<KeyPoint> pre_kpts;
    cv::Mat pre_des;
    for(int i = 0; i< n_pts.size(); i++)
    {
        int min = 3;
        KeyPoint kp;
        cv::Mat des;
        float maxResponse = 0;
        for(int ix = n_pts[i].x - min; ix < n_pts[i].x + min; ix++)
            for(int iy = n_pts[i].y - min; iy < n_pts[i].y + min; iy++)
            {
                if(ix<0) ix = 0;
                else if(ix>470) ix = 470;
                if(iy<0) iy = 0;
                else if(iy>630) iy = 630;

                int index = (int)indexMap.at<float>(ix,iy);
                if(index!=0)
                {   
                    //cout << i << "->"<< index << ": " << n_kpts[index].response << endl;
                    if(n_kpts[index].response > maxResponse)
                    {
                        kp = n_kpts[index];
                        des = n_des.rowRange(index,index+1).clone();
                        maxResponse = n_kpts[index].response;
                    } 
                }
            }
        if(maxResponse > 0)
        {
            pre_kpts.push_back(kp);
            pre_des.push_back(des);
        }
    }


    printf("t fast %lf ms\n", t_fast.toc());
    TicToc t_orb;
    vector<KeyPoint> forw_kpts;
    cv::Mat forw_des;
    ORB orb2(500,1.2,8, 31, 0, 2, ORB::HARRIS_SCORE, 31);
    orb2(gray2,cv::noArray(),forw_kpts,forw_des,false);
    //Ptr<ORB> orb2 = ORB::create(500,1.2f,8,31,0,2,ORB::FAST_SCORE,31,20);
    //orb2->detectAndCompute(gray2, cv::noArray(), forw_kpts,forw_des);
    
    printf("orb time %lf\n",t_orb.toc());
    //calculate bow of 2st frame
    TicToc t_bow2;
    vector<cv::Mat> vCurrentDesc2 = toDescriptorVector(forw_des);
    DBoW2::BowVector BowVec2;
    DBoW2::FeatureVector FeatVec2;
    mpVocabulary->transform(vCurrentDesc2,BowVec2,FeatVec2,4);
    printf("t bow2 %lf\n",t_bow2.toc());

    float score = mpVocabulary->score(BowVec1, BowVec2);
    printf("----------score = %f\n",score);

    BFMatcher matcher(NORM_HAMMING,true);
    std::vector< DMatch > matches;

    matcher.match( pre_des, forw_des, matches);
    printf("matches size %d \n",matches.size());
    sort(matches.begin(), matches.end(), [](const DMatch &a, const DMatch &b)
         {
             return a.distance < b.distance;
         });

    for (auto &p : forw_kpts)
    {
        Point2f tmp;
        tmp.x = (float)(int)p.pt.x;
        tmp.y = (float)(int)p.pt.y;
        cv::circle(rawImage2, tmp, 5, cv::Scalar(0,0,255));
    }
    //-- Draw matches
    cv::Mat rawImage21;
    cv::hconcat(rawImage2, rawImage1, rawImage21);
    
    for (int i = 0; i< 50&&i<matches.size(); i++)
    {
        
        Point2f forw_pt = forw_kpts[matches[i].trainIdx].pt;
        Point2f pre_pt = pre_kpts[matches[i].queryIdx].pt;
        pre_pt.x += 480.0;
        cv::line(rawImage21, forw_pt, pre_pt, cv::Scalar(0,255,0), 1, 8, 0);
        //cv::circle(rawImage21, forw_pts[i], 5, cv::Scalar(0,0,255));
        //cout << forw_pt  << pre_pt << endl;
        
        //printf("%d, %d %f\n",matches[i].queryIdx,matches[i].trainIdx, matches[i].distance);
    }

    namedWindow("3", CV_WINDOW_NORMAL);
    imshow("3",rawImage21);

    for (auto &p : pre_kpts)
    {
        Point2f tmp;
        tmp.x = (float)(int)p.pt.x;
        tmp.y = (float)(int)p.pt.y;
        cv::circle(rawImage1, tmp, 5, cv::Scalar(0,255,0));
    }
    
    for (auto &p : n_kpts)
    {
        Point2f tmp;
        tmp.x = (float)(int)p.pt.x;
        tmp.y = (float)(int)p.pt.y;
        cv::circle(rawImage1, tmp, 1, cv::Scalar(255,0,0));
    }
    printf("orb feature %d\n",n_kpts.size());
    namedWindow("2", CV_WINDOW_NORMAL);
    imshow("2",rawImage2);
    namedWindow("mask", CV_WINDOW_NORMAL);
    imshow("mask",rawImage1);
    
    /*
    vector<uchar> status;
    vector<float> err;
    cv::calcOpticalFlowPyrLK(gray1, gray2, n_pts, forw_pts, status, err, cv::Size(21, 21), 3);
    reduceVector(n_pts, status);
    reduceVector(forw_pts, status);

    rejectWithF();

    cv::Mat homoMatrix = Mat::eye(3,3,CV_32F);
    homoMatrix = findHomography(n_pts, forw_pts, CV_RANSAC, 3, noArray() );
    Mat alignImage2;
    warpPerspective (rawImage2, alignImage2, homoMatrix, rawImage1.size(),INTER_LINEAR + WARP_INVERSE_MAP);

    cv::Mat grayAlign2;
    cv::cvtColor(alignImage2, grayAlign2, CV_RGBA2GRAY);
    cv::calcOpticalFlowPyrLK(gray1, grayAlign2, pre_pts, forw_pts, status, err, cv::Size(21, 21), 3);
    reduceVector(pre_pts, status);
    reduceVector(forw_pts, status);

    cv::Mat rawImage21;
    cv::hconcat(alignImage2, rawImage1, rawImage21);
    for (int i = 0; i< forw_pts.size(); i++)
    {
        cv::line(rawImage21, forw_pts[i], cv::Point2f(pre_pts[i].x + 480, pre_pts[i].y), cv::Scalar(0,255,0), 1, 8, 0);
        cv::circle(rawImage21, forw_pts[i], 5, cv::Scalar(0,0,255));
    }
    TicToc t_blur;
    Mat downImage1,blurImage1;
    resize(gray1, downImage1, Size(0,0) , 1./12, 1./12, INTER_LINEAR );
    GaussianBlur(downImage1, blurImage1, Size(0,0), 2.0, 2.5, BORDER_DEFAULT );
    Mat downImage2,blurImage2;
    resize(gray2, downImage2, Size(0,0) , 1./12, 1./12, INTER_LINEAR );
    GaussianBlur(downImage2, blurImage2, Size(0,0), 2.0, 2.5, BORDER_DEFAULT );
    printf("blur time: %lfms\n", t_blur.toc());

    TicToc t_ECC;

    cout << homoMatrix << endl;
    Mat homoMatrix2 = Mat(3,3,CV_32F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
            homoMatrix2.at<float>(i, j) = (float)homoMatrix.at<double>(i, j);
    }
    cout << homoMatrix2 << endl;
    findTransformECC(downImage1, downImage2, homoMatrix2, MOTION_HOMOGRAPHY, TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001));
    warpPerspective (rawImage2, alignImage2, homoMatrix2, rawImage1.size(),INTER_LINEAR + WARP_INVERSE_MAP);
    printf("ecc time: %lfms\n", t_ECC.toc());

    imshow("1",rawImage21);
    namedWindow("2", CV_WINDOW_NORMAL);
    namedWindow("3", CV_WINDOW_NORMAL);
    imshow("2",alignImage2);
    imshow("3",blurImage2);
    //cv::waitkey(3000);

    */

    int i = 1;
    while(false)
    {
        stringstream ss;
		ss << i++;
		string filename = ss.str();
        std::string fileName = inputDirectory + "/" + filename + ".JPG";
  		std::cout << fileName << std::endl;
        cv::Mat rawImage = cv::imread(fileName.c_str());
        if(rawImage.data == NULL)
        {
            printf("Cannot Open Image\n");
            break;
        }
        else
        {
        	imshow("track",rawImage);
        	waitKey(1000);
        	//sleep(0.3);
        }
    }
    waitKey(-1);
    closedir(directory);
    
}
