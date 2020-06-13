//
// Created by pang on 2020/6/13.
//

#ifndef DWA_DEMO_DWA_H
#define DWA_DEMO_DWA_H
#include "utility.h"

class Dwa {
public:
    Dwa(const State& start, const Point& goal,  const Obstacle& obs, const Config& config);
    ~Dwa();
    bool stepOnceToGoal(std::vector<State>* best_trajectry, State* cur_state);

private:
    State motion(State x, Control u, float dt);
    Window calc_dynamic_window(State x, Config config);
    Traj calc_trajectory(State x, float v, float w, Config config);
    float calc_obstacle_cost(Traj traj, Obstacle ob, Config config);
    float calc_to_goal_cost(Traj traj, Point goal, Config config);
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
    State* dev_cur_x_;
    Point* dev_goal_;
    Point* dev_obs_;
    Control* dev_u_;
    Config* dev_config_;
    Window* dev_window_;

    State* dev_calulated_trajectories_;
    float* dev_costs_;



};


#endif //DWA_DEMO_DWA_H
