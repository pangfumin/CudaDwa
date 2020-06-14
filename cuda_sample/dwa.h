//
// Created by pang on 2020/6/13.
//

#ifndef DWA_DEMO_DWA_H
#define DWA_DEMO_DWA_H
#include "utility.h"

__host__ __device__
State motion(State x, Control u, float dt);

__host__ __device__
float calc_to_goal_cost(State last_traj_point, Point goal, Config config);

__host__ __device__
float calc_obstacle_cost_per_state(State x, Point* ob, int obs_cnt, Config config);

class Dwa {
public:
    Dwa(const State& start, const Point& goal,  const Obstacle& obs, const Config& config);
    ~Dwa();
    bool stepOnceToGoal(std::vector<State>* best_trajectry, State* cur_state);

private:
    Window calc_dynamic_window(State x, Config config);
    float calc_obstacle_cost(Traj traj, Obstacle ob, Config config);
    Traj calc_final_input(
            State x, Control& u,
            Window dw, Config config, Point goal,
            std::vector<Point>ob);

    int max_control_sample_cnt_;
    int motion_forward_steps_;

    // host
    Point goal_;
    Obstacle obs_;
    Config config_;
    State cur_x_;

    // dev
    Point* dev_obs_;

    Control* dev_control_samples_;
    State* dev_calulated_trajectories_;
    float* dev_costs_;
};


#endif //DWA_DEMO_DWA_H
