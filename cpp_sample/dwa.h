//
// Created by pang on 2020/6/13.
//

#ifndef DWA_DEMO_DWA_H
#define DWA_DEMO_DWA_H
#include "utility.h"

class Dwa {
public:
    Dwa(const State& start, const Point& goal,  const Obstacle& obs, const Config& config);

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



    Point goal_;
    Obstacle obs_;
    Config config_;

    State cur_x_;

};


#endif //DWA_DEMO_DWA_H
