#include "mbed.h"

class PID{
private :
    double _kp, _ki, _kd, _imax;
    //double prev_input;
    double P, I, D,  prev_D;
    double dt;
    double filter;

public:
    PID(double kp, double ki, double kd, double imax);
    double P_control(double error);
    double I_control(double error, double dt);
    double D_control(double input, double prev_input, double dt);
    double get_PID(double error, double input, double prev_input, double dt);
    void reset();
};
