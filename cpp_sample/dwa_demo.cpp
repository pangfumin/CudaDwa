/*************************************************************************
	> File Name: main.cpp
	> Author: TAI Lei
	> Mail: ltai@ust.hk
	> Created Time: Thu Mar  7 19:39:14 2019
 ************************************************************************/

#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include "utility.h"
#include "dwa.h"

cv::Point2i cv_offset(
        float x, float y, int image_width=2000, int image_height=2000){
    cv::Point2i output;
    output.x = int(x * 100) + image_width/2;
    output.y = image_height - int(y * 100) - image_height/3;
    return output;
};


int main(){
    State start{0.0, 0.0, PI/8.0, 0.0, 0.0};
    Point goal{10.0,10.0};
    Obstacle ob{
                        {-1, -1},
                        {0, 2},
                        {4.0, 2.0},
                        {5.0, 4.0},
                        {5.0, 5.0},
                        {5.0, 6.0},
                        {5.0, 9.0},
                        {8.0, 9.0},
                        {7.0, 9.0},
                        {12.0, 12.0}
                };

    Control u{0.0, 0.0};
    Config config;
    Traj traj;
    traj.push_back(start);

    bool terminal = false;

    cv::namedWindow("dwa", cv::WINDOW_NORMAL);
    int count = 0;

    Dwa dwa_demo(start, goal, ob, config);

    cv::Mat final_canvas;
    Traj ltraj;
    State x;
    while(!dwa_demo.stepOnceToGoal(&ltraj, &x)){

//        Traj ltraj = dwa_control(x, u, config, goal, ob);
//        x = motion(x, u, config.dt);
        traj.push_back(x);


        // visualization
        cv::Mat bg(3500,3500, CV_8UC3, cv::Scalar(255,255,255));
        cv::circle(bg, cv_offset(goal.x_, goal.y_, bg.cols, bg.rows),
                   30, cv::Scalar(255,0,0), 5);
        for(unsigned int j=0; j<ob.size(); j++){
            cv::circle(bg, cv_offset(ob[j].x_, ob[j].y_, bg.cols, bg.rows),
                       20, cv::Scalar(0,0,0), -1);
        }
        for(unsigned int j=0; j<ltraj.size(); j++){
            cv::circle(bg, cv_offset(ltraj[j].x_, ltraj[j].y_, bg.cols, bg.rows),
                       7, cv::Scalar(0,255,0), -1);
        }
        cv::circle(bg, cv_offset(x.x_, x.y_, bg.cols, bg.rows),
                   30, cv::Scalar(0,0,255), 5);


        cv::arrowedLine(
                bg,
                cv_offset(x.x_, x.y_, bg.cols, bg.rows),
                cv_offset(x.x_ + std::cos(x.theta_), x.y_ + std::sin(x.theta_), bg.cols, bg.rows),
                cv::Scalar(255,0,255),
                7);


        cv::imshow("dwa", bg);
        cv::waitKey(1);

        // std::string int_count = std::to_string(count);
        // cv::imwrite("./pngs/"+std::string(5-int_count.length(), '0').append(int_count)+".png", bg);

        count++;
        final_canvas = bg;
    }


    if (!final_canvas.empty()) {
        for(unsigned int j=0; j<traj.size(); j++){
            cv::circle(final_canvas, cv_offset(traj[j].x_, traj[j].y_, final_canvas.cols, final_canvas.rows),
                       7, cv::Scalar(0,0,255), -1);
        }

        cv::imshow("dwa", final_canvas);
        cv::waitKey();
    }

    return 0;



}