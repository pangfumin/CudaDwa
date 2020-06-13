//
// Created by pang on 2020/6/13.
//

#include "dwa.h"
#include<cmath>
#include<iostream>
#include<vector>
#include<array>
#include<cmath>
#include <limits>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )


void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

Dwa::Dwa(const State& start, const Point& goal,  const Obstacle& obs, const Config& config):
        cur_x_(start), goal_(goal), obs_(obs), config_(config)
{
    // alloc and copy
    checkCudaErrors( cudaMalloc( (void**)&dev_cur_x_, 1 * sizeof(State) ) );
    checkCudaErrors( cudaMemcpy( dev_cur_x_, &cur_x_, 1 * sizeof(State), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_goal_, 1 * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_goal_, &goal, 1 * sizeof(Point), cudaMemcpyHostToDevice ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_obs_, obs_.size() * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_obs_, &obs_, obs_.size() * sizeof(Point), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_u_, 1 * sizeof(Control) ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_config_, 1 * sizeof(Config) ) );
    checkCudaErrors( cudaMemcpy( dev_config_, &config, 1 * sizeof(Config), cudaMemcpyHostToDevice ) );

    int v_max_sample_cnt = 2 * (int)(config.max_accel * config.dt  / config.v_reso) + 1;
    int w_max_sample_cnt = 2 * (int)(config.max_dyawrate * config.dt  / config.yawrate_reso) + 1;

    max_control_sample_cnt_ = v_max_sample_cnt * w_max_sample_cnt;

//    std::cout << v_max_sample_cnt << " " << w_max_sample_cnt << " " << v_max_sample_cnt * w_max_sample_cnt << std::endl;

    motion_forward_steps_ = (int)(config.predict_time / config.dt) + 1;

    checkCudaErrors( cudaMalloc( (void**)&dev_calulated_trajectories_, max_control_sample_cnt_*motion_forward_steps_ * sizeof(State) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_costs_, max_control_sample_cnt_ * sizeof(float ) ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_window_, 1 * sizeof(Window) ) );


}

Dwa::~Dwa() {

    checkCudaErrors(cudaFree(dev_cur_x_));
    checkCudaErrors(cudaFree(dev_goal_));
    checkCudaErrors(cudaFree(dev_obs_));
    checkCudaErrors(cudaFree(dev_u_));
    checkCudaErrors(cudaFree(dev_config_));
    checkCudaErrors(cudaFree(dev_calulated_trajectories_));
    checkCudaErrors(cudaFree(dev_costs_));
    checkCudaErrors(cudaFree(dev_window_));


}


bool Dwa::stepOnceToGoal(std::vector<State>* best_trajectry, State* cur_state) {
    Control calculated_u;
    Window dw = calc_dynamic_window(cur_x_, config_);
    Traj ltraj = calc_final_input(cur_x_, calculated_u, dw, config_, goal_, obs_);

    cur_x_ = motion(cur_x_, calculated_u, config_.dt);

    //
    *best_trajectry = ltraj;
    *cur_state = cur_x_;
    if (std::sqrt(std::pow((cur_x_.x_ - goal_.x_), 2) + std::pow((cur_x_.y_ - goal_.y_), 2)) <= config_.robot_radius){
       return true;
    }
    return false;
}


State Dwa::motion(State x, Control u, float dt){
    x.theta_ += u.w_ * dt;
    x.x_ += u.v_ * std::cos(x.theta_) * dt;
    x.y_ += u.v_ * std::sin(x.theta_) * dt;
    x.v_ = u.v_;
    x.w_ = u.w_;
    return x;
};

Window Dwa::calc_dynamic_window(State x, Config config){

    return {
            std::max((x.v_ - config.max_accel * config.dt), config.min_speed),
            std::min((x.v_ + config.max_accel * config.dt), config.max_speed),
            std::max((x.w_ - config.max_dyawrate * config.dt), -config.max_yawrate),
            std::min((x.w_ + config.max_dyawrate * config.dt), config.max_yawrate)
    };

};


Traj Dwa::calc_trajectory(State x, float v, float w, Config config){

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


float Dwa::calc_obstacle_cost(Traj traj, Obstacle ob, Config config){
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

float Dwa::calc_to_goal_cost(Traj traj, Point goal, Config config){

    float goal_magnitude = std::sqrt(goal.x_*goal.x_ + goal.y_*goal.y_);
    float traj_magnitude = std::sqrt(std::pow(traj.back().x_, 2) + std::pow(traj.back().y_, 2));
    float dot_product = (goal.x_ * traj.back().x_) + (goal.y_ * traj.back().y_);
    float error = dot_product / (goal_magnitude * traj_magnitude);
    float error_angle = std::acos(error);
    float cost = config.to_goal_cost_gain * error_angle;

    return cost;
};

Traj Dwa::calc_final_input(
        State x, Control& u,
        Window dw, Config config, Point goal,
        std::vector<Point>ob){

    float min_cost = 10000.0;

    Traj best_traj;

    std::vector<Control> control_samples(max_control_sample_cnt_);
    std::vector<float> final_costs(max_control_sample_cnt_);
    std::vector<Traj> trajecotries(max_control_sample_cnt_);

    int valid_control_sample_cnt = 0;
    for (float v=dw.min_v_; v<=dw.max_v_; v+=config.v_reso){
        for (float w=dw.min_w_; w<=dw.max_w_; w+=config.yawrate_reso) {
            control_samples.at(valid_control_sample_cnt) = Control{v,w};
            valid_control_sample_cnt ++;
        }
    }

    std::cout << "valid_control_sample_cnt: " << valid_control_sample_cnt << " " << max_control_sample_cnt_ << std::endl;


    // todo: parallelization
    for (int i = 0; i < valid_control_sample_cnt; i++) {
        auto vw = control_samples[i];

        Traj traj = calc_trajectory(x, vw.v_, vw.w_, config);

        float to_goal_cost = calc_to_goal_cost(traj, goal, config);
        float speed_cost = config.speed_cost_gain * (config.max_speed - traj.back().v_);
        float ob_cost = calc_obstacle_cost(traj, ob, config);
        float final_cost = to_goal_cost + speed_cost + ob_cost;

        final_costs.at(i) = final_cost;
        trajecotries.at(i) = traj;

    }

    int best_idx = 0;
    for (int i = 0; i < valid_control_sample_cnt; i++) {
        if (final_costs[i] < min_cost) {
            min_cost = final_costs[i];
            best_idx = i;
        }
    }


    u =  control_samples[best_idx];
    best_traj = trajecotries.at(best_idx);

    return best_traj;
};



