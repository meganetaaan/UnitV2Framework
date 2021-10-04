#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "net.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <thread>
#include <queue>
#include <mutex>
#include <unistd.h> // TODO: ?
#include "base64.h"
#include "benchmark.h"
#include "ArduinoJson.h"
#include "framework.h"

#define IMAGE_DIV 2

enum
{
    IMAGE_FEED_MODE_RGB,
    IMAGE_FEED_MODE_MASK
};

int image_feed_mode = IMAGE_FEED_MODE_RGB;

uint8_t l_min = 0, l_max = 27;
uint8_t a_min = 108, a_max = 148;
uint8_t b_min = 108, b_max = 148;

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

#define SEGMENT_NUM     (8)
#define SEGMENT_HEIGHT  ((480 / IMAGE_DIV) / 8)
#define WINDOW_WIDTH    (100 / IMAGE_DIV)
#define WINDOW_WIDTH_DIV2   (WINDOW_WIDTH / 2)

int segment_pos[SEGMENT_NUM][2] = {0};
std::vector<cv::Point> segment_point;
int segment_pos_avg = 0;
int segment_pos_sum = 0;
int computeSegment(cv::Mat &src, cv::Mat &image, int seg, int left_x, int right_x, int &next_left_x, int &next_right_x)
{
    int lrange, rrange, idx;
    int max_hist;
cv::Point marker;
    int urange = seg * SEGMENT_HEIGHT;
    int brange = seg * SEGMENT_HEIGHT + SEGMENT_HEIGHT;

    lrange = left_x - WINDOW_WIDTH_DIV2;
    rrange = left_x + WINDOW_WIDTH_DIV2;
    if (lrange < 0)
    {
        lrange = 0;
    }
    if (rrange > (640 / IMAGE_DIV))
    {
        rrange = 640 / IMAGE_DIV;
    }

    max_hist = 0;
    idx = 0;
    int left_suma = 0;
    int left_sumb = 0;
    int left_vhist[WINDOW_WIDTH];

    marker.x = left_x;
    marker.y = (seg * SEGMENT_HEIGHT) + (SEGMENT_HEIGHT / 2);
    #ifdef LOCAL_RENDER
        cv::drawMarker(src, marker, cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 10, 2, 8);
    #else
        RD_addPoint(marker.x, marker.y, "#CDCD00", IMAGE_DIV);
    #endif

    // バイナリ画像を走査してヒストグラムを作っている
    for(int i = lrange; i < rrange; i++)
    {
        left_vhist[idx] = 0;
        for(int j = urange; j < brange; j++)
        {
            if(image.at<uint8_t>(j, i))
            {
                left_vhist[idx]++;
            }
        }
        left_suma += left_vhist[idx] * idx;
        left_sumb += left_vhist[idx];
        idx++;
    }

    // 次に走査するセグメントの決定
    if(left_sumb == 0)
    {
        next_left_x = left_x;
    }
    else
    {
        next_left_x = left_suma / left_sumb + lrange;
    }
    segment_pos[seg][0] = next_left_x;

    lrange = right_x - WINDOW_WIDTH_DIV2;
    rrange = right_x + WINDOW_WIDTH_DIV2;

    if(lrange < 0)
    {
        lrange = 0;
    }
    if(rrange > (640 / IMAGE_DIV))
    {
        rrange = 640 / IMAGE_DIV;
    }

    max_hist = 0;
    idx = 0;
    int right_suma = 0;
    int right_sumb = 0;
    int right_vhist[WINDOW_WIDTH];

    marker.x = right_x;
    #ifdef LOCAL_RENDER
        cv::drawMaker(src,marker,cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 10, 2, 8);
    #else
        RD_addPoint(marker.x, marker.y, "#CDCD00", IMAGE_DIV);
    #endif

    for(int i = lrange; i < rrange; i++)
    {
        right_vhist[idx] = 0;
        for(int j = urange; j < brange; j++)
        {
            if(image.at<uint8_t>(j, 8))
            {
                right_vhist[idx]++;
            }
        }
        right_suma += right_vhist[idx] * idx;
        right_sumb += right_vhist[idx];
        idx++;
    }
    
    if(right_sumb == 0)
    {
        next_right_x = right_x;
    }
    else
    {
        next_right_x = right_suma / right_sumb + lrange;
    }
    segment_pos[seg][1] = next_right_x;
    cv::Point p;
    p.x = next_left_x + ((next_right_x - next_left_x) / 2);
    p.y = (seg * SEGMENT_HEIGHT) + (SEGMENT_HEIGHT / 2);
    segment_point.push_back(p);
    segment_pos_sum += p.x;

    return right_suma < right_sumb ? right_suma : right_sumb;
}

void findLine(cv::Mat &image_mask, int &left_x, int &right_x)
{
    int vhist[image_mask.cols];

    for(int i = 0; i < image_mask.cols; i++)
    {
        vhist[i] = 0;
        for(int j = 0; j < image_mask.rows; j++)
        {
            if(image_mask.at<uint8_t>(j, i))
            {
                vhist[i]++;
            }
        }
    }

    int left_cnt = 0;
    for(int i = 0; i < image_mask.cols / 2; i++)
    {
        if(vhist[i] > left_cnt)
        {
            left_cnt = vhist[i];
            left_x = i;
        }
    }

    int right_cnt = 0;
    for(int i = image_mask.cols / 2; i < image_mask.cols; i++)
    {
        if(vhist[i] > right_cnt)
        {
            right_cnt = vhist[i];
            right_x = i;
        }
    }
}

int left_x = 0, right_x = 0;
void process(cv::Mat &image)
{
    int next_left_x;
    int next_right_x;
    int input_left_x = left_x;
    int input_right_x = right_x;

    cv::Mat image_lab;
    cv::Mat image_mask;
    cv::cvtColor(image, image_lab, cv::COLOR_BGR2Lab);
    cv::inRange(image_lab, cv::Scalar(l_min, a_min, b_min), cv::Scalar(l_max, a_max, b_max), image_mask);
    cv::morphologyEx(image_mask, image_mask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

}