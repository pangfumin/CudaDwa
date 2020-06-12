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

#include "dwa.h"


using Traj = std::vector<State>;

using Obstacle = std::vector<Point>;;


State motion(State x, Control u, float dt){
    x.theta_ += u.w_ * dt;
    x.x_ += u.v_ * std::cos(x.theta_) * dt;
    x.y_ += u.v_ * std::sin(x.theta_) * dt;
    x.v_ = u.v_;
    x.w_ = u.w_;
    return x;
};

Window calc_dynamic_window(State x, Config config){

    return {
                    std::max((x.v_ - config.max_accel * config.dt), config.min_speed),
                    std::min((x.v_ + config.max_accel * config.dt), config.max_speed),
                    std::max((x.w_ - config.max_dyawrate * config.dt), -config.max_yawrate),
                    std::min((x.w_ + config.max_dyawrate * config.dt), config.max_yawrate)
            };
};


Traj calc_trajectory(State x, float v, float w, Config config){

    Traj traj;
    traj.push_back(x);
    float time = 0.0;
    while (time <= config.predict_time){
        x = motion(x, Control{v, w}, config.dt);
        traj.push_back(x);
        time += config.dt;
    }
    return traj;
};


float calc_obstacle_cost(Traj traj, Obstacle ob, Config config){
    // calc obstacle cost inf: collistion, 0:free
    int skip_n = 2;
    float minr = std::numeric_limits<float>::max();

    for (unsigned int ii=0; ii<traj.size(); ii+=skip_n){
        for (unsigned int i=0; i< ob.size(); i++){
            float ox = ob[i].x_;
            float oy = ob[i].y_;
            float dx = traj[ii].x_ - ox;
            float dy = traj[ii].y_ - oy;

            float r = std::sqrt(dx*dx + dy*dy);
            if (r <= config.robot_radius){
                return std::numeric_limits<float>::max();
            }

            if (minr >= r){
                minr = r;
            }
        }
    }

    return 1.0 / minr;
};

float calc_to_goal_cost(Traj traj, Point goal, Config config){

    float goal_magnitude = std::sqrt(goal.x_*goal.x_ + goal.y_*goal.y_);
    float traj_magnitude = std::sqrt(std::pow(traj.back().x_, 2) + std::pow(traj.back().y_, 2));
    float dot_product = (goal.x_ * traj.back().x_) + (goal.y_ * traj.back().y_);
    float error = dot_product / (goal_magnitude * traj_magnitude);
    float error_angle = std::acos(error);
    float cost = config.to_goal_cost_gain * error_angle;

    return cost;
};

Traj calc_final_input(
        State x, Control& u,
        Window dw, Config config, Point goal,
        std::vector<Point>ob){

    float min_cost = 10000.0;
    Control min_u = u;
    min_u.v_ = 0.0;
    Traj best_traj;

    // evalucate all trajectory with sampled input in dynamic window
    int traj_cnt = 0;
    for (float v=dw.min_v_; v<=dw.max_v_; v+=config.v_reso){
        for (float w=dw.min_w_; w<=dw.max_w_; w+=config.yawrate_reso){

            Traj traj = calc_trajectory(x, v, w, config);

            float to_goal_cost = calc_to_goal_cost(traj, goal, config);
            float speed_cost = config.speed_cost_gain * (config.max_speed - traj.back().v_);
            float ob_cost = calc_obstacle_cost(traj, ob, config);
            float final_cost = to_goal_cost + speed_cost + ob_cost;

            if (min_cost >= final_cost){
                min_cost = final_cost;
                min_u = Control{v, w};
                best_traj = traj;
            }
            traj_cnt ++;
        }
    }

    std::cout << "traj_cnt: " << traj_cnt << std::endl;
    u = min_u;
    return best_traj;
};


Traj dwa_control(State x, Control & u, Config config,
                 Point goal, Obstacle ob){
    // # Dynamic Window control
    Window dw = calc_dynamic_window(x, config);
    Traj traj = calc_final_input(x, u, dw, config, goal, ob);

    return traj;
}

cv::Point2i cv_offset(
        float x, float y, int image_width=2000, int image_height=2000){
    cv::Point2i output;
    output.x = int(x * 100) + image_width/2;
    output.y = image_height - int(y * 100) - image_height/3;
    return output;
};


int main(){
    State x{0.0, 0.0, PI/8.0, 0.0, 0.0};
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
    traj.push_back(x);

    bool terminal = false;

    cv::namedWindow("dwa", cv::WINDOW_NORMAL);
    int count = 0;


    cv::Mat final_canvas;
    for(int i=0; i<10000 && !terminal; i++){
        Traj ltraj = dwa_control(x, u, config, goal, ob);
        x = motion(x, u, config.dt);
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

        if (std::sqrt(std::pow((x.x_ - goal.x_), 2) + std::pow((x.y_ - goal.y_), 2)) <= config.robot_radius){
            terminal = true;
            final_canvas = bg;
        }




        cv::imshow("dwa", bg);
        cv::waitKey(1);

        // std::string int_count = std::to_string(count);
        // cv::imwrite("./pngs/"+std::string(5-int_count.length(), '0').append(int_count)+".png", bg);

        count++;
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