//
// Created by pang on 2020/6/13.
//

#ifndef DWA_DEMO_UTILITY_H
#define DWA_DEMO_UTILITY_H

#include <vector>


#define PI 3.141592653



struct State {
    float x_;
    float y_;
    float theta_;
    float v_;
    float w_;
};


using Traj = std::vector<State>;

struct Window {
    float min_v_;
    float max_v_;
    float min_w_;
    float max_w_;

};

struct Control {
    float v_;
    float w_;
};

struct Point {
    float x_;
    float y_;
};

using Obstacle = std::vector<Point>;;

struct Config{
    float max_speed = 1.0;
    float min_speed = -0.5;
    float max_yawrate = 40.0 * PI / 180.0;
    float max_accel = 0.2;
    float robot_radius = 1.0;
    float max_dyawrate = 40.0 * PI / 180.0;

    float v_reso = 0.01;
    float yawrate_reso = 0.01 * PI / 180.0;

    float dt = 0.1;
    float predict_time = 3.0;
    float to_goal_cost_gain = 1.0;
    float speed_cost_gain = 1.0;
};



#endif //DWA_DEMO_UTILITY_H
