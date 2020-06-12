#include "dwa.h"


//#include <cutil_inline.h>
//#include <cutil_math.h>
#include <cuda_runtime.h>

__host__ __device__
State motion_host_device(State x, Control u, float dt){
    x.theta_ += u.w_ * dt;
    x.x_ += u.v_ * std::cos(x.theta_) * dt;
    x.y_ += u.v_ * std::sin(x.theta_) * dt;
    x.v_ = u.v_;
    x.w_ = u.w_;
    return x;
}



extern "C" State* calc_final_input(
        State* x, Control* u,
        Window* dw, Config* config, Point* goal,
        Point* ob) {
    State* ptr;
    return ptr;
}

extern "C"  State motion(State x, Control u, float dt){

    return motion_host_device(x,u,dt);
}

