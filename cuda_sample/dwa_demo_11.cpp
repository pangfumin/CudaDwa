
#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

#include "dwa.h"
#include <cuda_runtime_api.h>
#include <cuda.h>







using Traj = std::vector<State>;

using Obstacle = std::vector<Point>;;


extern "C"  State motion(State x, Control u, float dt);
extern "C" State* calc_final_input(
        State* x, Control* u,
        Window* dw, Config* config, Point* goal,
        Point* ob);

Window calc_dynamic_window(State x, Config config){

    return {
                    std::max((x.v_ - config.max_accel * config.dt), config.min_speed),
                    std::min((x.v_ + config.max_accel * config.dt), config.max_speed),
                    std::max((x.w_ - config.max_dyawrate * config.dt), -config.max_yawrate),
                    std::min((x.w_ + config.max_dyawrate * config.dt), config.max_yawrate)
            };


};


Traj calc_trajectory(State x, float v, float w, int forward_staps, Config config){

    Traj traj;
    traj.push_back(x);
    float time = 0.0;
    while (forward_staps --){
        x = motion(x, Control{v, w}, config.dt);
        traj.push_back(x);
        time += config.dt;
    }
//    std::cout << "traje: " << traj.size() << std::endl;
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
    int traj_v_cnt = 0;
    int traj_w_cnt = 0;
    int traj_cnt = 0;

    for (float v=dw.min_v_; v<=dw.max_v_; v+=config.v_reso){
        traj_v_cnt++;
        traj_w_cnt = 0;
        for (float w=dw.min_w_; w<=dw.max_w_; w+=config.yawrate_reso){

            int forward_steps = (int)(config.predict_time / config.dt);
            Traj traj = calc_trajectory(x, v, w, forward_steps, config);

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
            traj_w_cnt ++;

        }


    }

    std::cout << "traj_cnt: "<< traj_cnt << " "  <<  traj_v_cnt << " " << traj_w_cnt << std::endl;

//    std::cout << "traj_cnt: " << traj_cnt << std::endl;
    u = min_u;
    return best_traj;
};



cv::Point2i cv_offset(
        float x, float y, int image_width=2000, int image_height=2000){
    cv::Point2i output;
    output.x = int(x * 100) + image_width/2;
    output.y = image_height - int(y * 100) - image_height/3;
    return output;
};


int main(){
    State host_x{0.0, 0.0, PI/8.0, 0.0, 0.0};

    Point host_goal{10.0,10.0};
    Obstacle host_ob{
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

    Control host_u{0.0, 0.0};
    Config host_config;
    Traj traj;
    traj.push_back(host_x);

    bool terminal = false;

    cv::namedWindow("dwa", cv::WINDOW_NORMAL);
    int count = 0;

    State* dev_x;
    Point* dev_goal;
    Point* dev_ob;
    Control* dev_u;
    Config* dev_config;
    Window* dev_window;

    State* dev_calulated_trajectories;
    float* dev_costs;


    checkCudaErrors( cudaMalloc( (void**)&dev_x, 1 * sizeof(State) ) );
    checkCudaErrors( cudaMemcpy( dev_x, &host_x, 1 * sizeof(State), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_goal, 1 * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_goal, &host_goal, 1 * sizeof(Point), cudaMemcpyHostToDevice ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_ob, host_ob.size() * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_ob, &host_ob, host_ob.size() * sizeof(Point), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_u, 1 * sizeof(Control) ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_config, 1 * sizeof(Config) ) );
    checkCudaErrors( cudaMemcpy( dev_config, &host_config, 1 * sizeof(Config), cudaMemcpyHostToDevice ) );


    int v_max_sample_cnt = 2 * (int)(host_config.max_accel * host_config.dt  / host_config.v_reso) + 1;
    int w_max_sample_cnt = 2 * (int)(host_config.max_dyawrate * host_config.dt  / host_config.yawrate_reso) + 1;

    std::cout << v_max_sample_cnt << " " << w_max_sample_cnt << " " << v_max_sample_cnt * w_max_sample_cnt << std::endl;

    int motion_forward_steps = (int)(host_config.predict_time / host_config.dt) + 1;

    checkCudaErrors( cudaMalloc( (void**)&dev_calulated_trajectories, v_max_sample_cnt*w_max_sample_cnt*motion_forward_steps * sizeof(State) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_costs, v_max_sample_cnt*w_max_sample_cnt * sizeof(float ) ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_window, 1 * sizeof(Window) ) );

    cv::Mat final_canvas;
    for(int i=0; i<10000 && !terminal; i++){
        Window dw = calc_dynamic_window(host_x, host_config);
        checkCudaErrors( cudaMemcpy( dev_window, &dw, 1 * sizeof(Window), cudaMemcpyHostToDevice ) );
         calc_final_input(
                dev_x, dev_u,
                dev_window, dev_config, dev_goal,
                dev_ob);

        Traj ltraj = calc_final_input(host_x, host_u, dw, host_config, host_goal, host_ob);

        host_x = motion(host_x, host_u, host_config.dt);
        traj.push_back(host_x);


        // visualization
        cv::Mat bg(3500,3500, CV_8UC3, cv::Scalar(255,255,255));
        cv::circle(bg, cv_offset(host_goal.x_, host_goal.y_, bg.cols, bg.rows),
                   30, cv::Scalar(255,0,0), 5);
        for(unsigned int j=0; j<host_ob.size(); j++){
            cv::circle(bg, cv_offset(host_ob[j].x_, host_ob[j].y_, bg.cols, bg.rows),
                       20, cv::Scalar(0,0,0), -1);
        }
        for(unsigned int j=0; j<ltraj.size(); j++){
            cv::circle(bg, cv_offset(ltraj[j].x_, ltraj[j].y_, bg.cols, bg.rows),
                       7, cv::Scalar(0,255,0), -1);
        }
        cv::circle(bg, cv_offset(host_x.x_, host_x.y_, bg.cols, bg.rows),
                   30, cv::Scalar(0,0,255), 5);


        cv::arrowedLine(
                bg,
                cv_offset(host_x.x_, host_x.y_, bg.cols, bg.rows),
                cv_offset(host_x.x_ + std::cos(host_x.theta_), host_x.y_ + std::sin(host_x.theta_), bg.cols, bg.rows),
                cv::Scalar(255,0,255),
                7);

        if (std::sqrt(std::pow((host_x.x_ - host_goal.x_), 2) + std::pow((host_x.y_ - host_goal.y_), 2)) <= host_config.robot_radius){
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

    checkCudaErrors(cudaFree(dev_x));
    checkCudaErrors(cudaFree(dev_goal));
    checkCudaErrors(cudaFree(dev_ob));
    checkCudaErrors(cudaFree(dev_u));
    checkCudaErrors(cudaFree(dev_config));
    checkCudaErrors(cudaFree(dev_calulated_trajectories));
    checkCudaErrors(cudaFree(dev_costs));
    checkCudaErrors(cudaFree(dev_window));

    return 0;



}