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


#define imin(a,b) (a<b?a:b)
const int N = 6 * 1024 ;
const int threadsPerBlock = 16;
const int blocksPerGrid =
        (N+threadsPerBlock-1) / threadsPerBlock ;


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
    int v_max_sample_cnt = 2 * (int)(config.max_accel * config.dt  / config.v_reso) + 1;
    int w_max_sample_cnt = 2 * (int)(config.max_dyawrate * config.dt  / config.yawrate_reso) + 1;

    max_control_sample_cnt_ = v_max_sample_cnt * w_max_sample_cnt;

    motion_forward_steps_ = (int)(config.predict_time / config.dt) + 1;


    // constant
    checkCudaErrors( cudaMalloc( (void**)&dev_goal_, 1 * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_goal_, &goal, 1 * sizeof(Point), cudaMemcpyHostToDevice ) );

    checkCudaErrors( cudaMalloc( (void**)&dev_obs_, obs_.size() * sizeof(Point) ) );
    checkCudaErrors( cudaMemcpy( dev_obs_, obs_.data(), obs_.size() * sizeof(Point), cudaMemcpyHostToDevice ) );

    int obs_cnt =  obs_.size();
    checkCudaErrors( cudaMalloc( (void**)&dev_obs_cnt_, 1 * sizeof(int) ) );
    checkCudaErrors( cudaMemcpy( dev_obs_cnt_, &obs_cnt, 1 * sizeof(int), cudaMemcpyHostToDevice ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_config_, 1 * sizeof(Config) ) );
    checkCudaErrors( cudaMemcpy( dev_config_, &config, 1 * sizeof(Config), cudaMemcpyHostToDevice ) );


    checkCudaErrors( cudaMalloc( (void**)&dev_motion_forward_steps_, 1 * sizeof(int) ) );
    checkCudaErrors( cudaMemcpy( dev_motion_forward_steps_, &motion_forward_steps_, 1 * sizeof(int), cudaMemcpyHostToDevice ) );

    // variable
    checkCudaErrors( cudaMalloc( (void**)&dev_cur_x_, 1 * sizeof(State) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_control_samples_, max_control_sample_cnt_ * sizeof(Control) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_calulated_trajectories_, max_control_sample_cnt_*motion_forward_steps_ * sizeof(State) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_costs_, max_control_sample_cnt_ * sizeof(float ) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_window_, 1 * sizeof(Window) ) );
    checkCudaErrors( cudaMalloc( (void**)&dev_valid_control_sample_cnt_, 1 * sizeof(int) ) );

}

Dwa::~Dwa() {

    checkCudaErrors(cudaFree(dev_cur_x_));
    checkCudaErrors(cudaFree(dev_goal_));
    checkCudaErrors(cudaFree(dev_obs_));
    checkCudaErrors(cudaFree(dev_config_));
    checkCudaErrors(cudaFree(dev_calulated_trajectories_));
    checkCudaErrors(cudaFree(dev_costs_));
    checkCudaErrors(cudaFree(dev_window_));
    checkCudaErrors(cudaFree(dev_valid_control_sample_cnt_));
    checkCudaErrors(cudaFree(dev_motion_forward_steps_));
    checkCudaErrors(cudaFree(dev_obs_cnt_));
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

__host__ __device__
 State motion(State x, Control u, float dt){
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

__host__ __device__
float calc_obstacle_cost_per_state(State x, Point* ob, int obs_cnt, Config config){
    // calc obstacle cost inf: collistion, 0:free
    float minr = std::numeric_limits<float>::max();
    for (unsigned int i = 0; i < obs_cnt; i++){
        float ox = ob[i].x_;
        float oy = ob[i].y_;
        float dx = x.x_ - ox;
        float dy = x.y_ - oy;
        float r = std::sqrt(dx*dx + dy*dy);

        if (r <= config.robot_radius){
            return std::numeric_limits<float>::max();
        }

        if (minr >= r){
            minr = r;
        }
    }

    return 1.0 / minr;
};


__host__ __device__
float calc_to_goal_cost(State last_traj_point, Point goal, Config config){

    float goal_magnitude = std::sqrt(goal.x_*goal.x_ + goal.y_*goal.y_);
    float traj_magnitude = std::sqrt(std::pow(last_traj_point.x_, 2) + std::pow(last_traj_point.y_, 2));
    float dot_product = (goal.x_ * last_traj_point.x_) + (goal.y_ * last_traj_point.y_);
    float error = dot_product / (goal_magnitude * traj_magnitude);
    float error_angle = std::acos(error);
    float cost = config.to_goal_cost_gain * error_angle;

    return cost;
};



__global__ void calc_trajectories_costs_kernel(
         State* cur_state,
         Point* goal,
         Point* obs,
         Config* config,
         Control* control_samples,
         State* calculated_trajectries,
         float* costs,
         int *valid_sample_cnt, int  *motion_steps,
         int *obs_cnt) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int trajectory_idx = tid * (*motion_steps);
    int index =  trajectory_idx;

    if (tid < *valid_sample_cnt) {
        Control vw = control_samples[tid];
        State tem = *cur_state;
        int steps = *motion_steps;
        float to_goal_cost = 0;
        float speed_cost = 0;
        float ob_cost = 0;
        while (steps){
            tem = motion(tem, vw, config->dt);
            calculated_trajectries[index] = tem;
            index ++;

            float ob_cost_per_state =  calc_obstacle_cost_per_state(tem, obs, *obs_cnt, *config);
            if (ob_cost < ob_cost_per_state) {
                ob_cost = ob_cost_per_state;
            }

            if (steps == 1) {
                to_goal_cost = calc_to_goal_cost(tem, *goal, *config);
                speed_cost = config->speed_cost_gain * (config->max_speed - tem.v_);

            }
            steps --;
        }


        float final_cost = to_goal_cost + speed_cost + ob_cost;

        costs[tid] = final_cost;
    }
}


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

//    std::cout << "valid_control_sample_cnt: " << valid_control_sample_cnt << " " << max_control_sample_cnt_ << std::endl;


    cudaEvent_t     gpu_start, gpu_stop;
    checkCudaErrors( cudaEventCreate( &gpu_start ) );
    checkCudaErrors( cudaEventCreate( &gpu_stop ) );
    checkCudaErrors( cudaEventRecord( gpu_start, 0 ) );

    checkCudaErrors( cudaMemcpy( dev_cur_x_, &cur_x_, 1 * sizeof(State), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( dev_valid_control_sample_cnt_, &valid_control_sample_cnt, 1 * sizeof(int), cudaMemcpyHostToDevice ) );
    checkCudaErrors( cudaMemcpy( dev_control_samples_, control_samples.data(), control_samples.size() * sizeof(Control), cudaMemcpyHostToDevice ) );

    calc_trajectories_costs_kernel<<<blocksPerGrid,threadsPerBlock>>>(
            dev_cur_x_, //State* cur_state,
            dev_goal_,// Point* goal,
            dev_obs_,//Point* obs,
            dev_config_,
            dev_control_samples_, //Control* control_samples,
            dev_calulated_trajectories_,//State* calculated_trajectries,
            dev_costs_,//float* costs,
            dev_valid_control_sample_cnt_,//int *valid_sample_cnt,
            dev_motion_forward_steps_, //int  *motion_steps,
            dev_obs_cnt_ //int *obs_cnt
            );





    // fetch const
    std::vector<float> gpu_final_costs(max_control_sample_cnt_);
    checkCudaErrors( cudaMemcpy( gpu_final_costs.data(), dev_costs_, control_samples.size() * sizeof(float), cudaMemcpyDeviceToHost ) );

    // find the best one
    float gpu_min_cost = 10000.0;;
    int gpu_best_idx;
    for (int i = 0; i < valid_control_sample_cnt; i++) {
        if (gpu_final_costs[i] < gpu_min_cost) {
            gpu_min_cost = gpu_final_costs[i];
            gpu_best_idx = i;
        }
    }

    // fetch best trajectry
    std::vector<State> gpu_best_traj(motion_forward_steps_);
    checkCudaErrors( cudaMemcpy( gpu_best_traj.data(),
            dev_calulated_trajectories_ + gpu_best_idx * motion_forward_steps_,
                                 motion_forward_steps_ * sizeof(State), cudaMemcpyDeviceToHost ) );


    // get stop time, and display the timing results
    checkCudaErrors( cudaEventRecord( gpu_stop, 0 ) );
    checkCudaErrors( cudaEventSynchronize( gpu_stop ) );
    float   GPU_elapsedTime;
    checkCudaErrors( cudaEventElapsedTime( &GPU_elapsedTime,
                                           gpu_start, gpu_stop ) );
    printf( "GPU Time to process:  %3.1f ms\n", GPU_elapsedTime );


#define ALSO_CALCULATE_CPU
#ifdef ALSO_CALCULATE_CPU

    // capture the start time
    clock_t         cpu_start, cpu_stop;
    cpu_start = clock();

    for (int i = 0; i < valid_control_sample_cnt; i++) {
        auto vw = control_samples[i];

        Traj traj;
        auto tem = x;
        int steps = motion_forward_steps_;
        while (steps ){
            tem = motion(tem, vw, config.dt);
            traj.push_back(tem);

            steps --;
        }


        float to_goal_cost = calc_to_goal_cost(traj.back(), goal, config);
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

    cpu_stop = clock();
    float   CPU_elapsedTime = (float)(cpu_stop - cpu_start) /
                          (float)CLOCKS_PER_SEC * 1000.0f;
    printf( "CPU Time to process:  %3.1f ms\n", CPU_elapsedTime );

    std::cout << "best_idx: " << best_idx << " " << gpu_best_idx << std::endl;
#endif

    u =  control_samples[gpu_best_idx];
    best_traj = gpu_best_traj;

    return best_traj;
};



