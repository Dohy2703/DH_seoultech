#include "mbed.h"
#include "PID.h"

PID::PID(double kp, double ki, double kd, double imax){
    _kp=kp;
    _ki=ki;
    _kd=kd;
    _imax=imax;
}


double PID::P_control(double error){
    P= error *_kp;
    return P;
}

double PID::I_control(double error, double dt){
    I+=(error*_ki)*dt;
    if(I>_imax) I=_imax;
    else if(I<-_imax) I=_imax;
    return I;
}

double PID::D_control(double input,double prev_input, double dt){
    D=(input-prev_input)/dt;
    prev_input=input;
    prev_D=D;
    
    return D*_kd;
}

double PID::get_PID(double error, double input,double prev_input, double dt){
     return P_control(error)+I_control(error, dt)-D_control(input,prev_input, dt);
}

void PID::reset(){
    I=0;
    //prev_input=0;
    prev_D=0;

}
