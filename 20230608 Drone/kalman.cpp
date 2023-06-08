#include "kalman.h"
// derived from http://blog.tkjelectronics.dk/2012/09/a-practical-approach-to-kalman-filter-and-how-to-implement-it/
kalman::kalman(void){
    P[0][0]     = 0;
    P[0][1]     = 0;
    P[1][0]     = 0;
    P[1][1]     = 0;
    
    dist       = 0;
    dist_p     = 0;

    vel        = 0;
    vel_p      = 0;
    //Q_dist     = 0.1;
    //Q_velBias  = 3;
   // R_dist     = 1000;
    Q_dist = 0.004; // 0.0004 //0.004
    Q_velBias = 0.08; //0 .1//0.05
    R_dist=15.0;  // 1   0.1 //10
    // Q_dist = 0.004;
    // Q_velBias = 1;
    // R_dist = 36;
}

double kalman::getdist(double newdist, double dt){
    //predict the dist according to the velscope
    vel_p         = vel;
    dist_p        = dist+ dt* vel;
    //update the error covariancesf
    P[0][0]     += dt *(P[1][0]+P[0][1]+P[1][1]*dt)+Q_dist;
    P[0][1]     += dt * P[1][1];
    P[1][0]     += dt * P[1][1];
    P[1][1]     += Q_velBias;
    //difference between the predicted dist and the accelerometer dist
    y            = newdist - dist_p;
    //estimation error
    S            = P[1][0] + R_dist;
    //determine the kalman gain according to the error covariance and the distrust
    K[0]         = P[0][0]/S;
    K[1]         = P[1][0]/S;
    //adjust the dist according to the kalman gain and the difference with the measurement
    dist_p       += K[0] * y;
    vel_p        += K[1] * y;
    //update the error covariance
    P[0][0]     -= K[0] * P[0][0];
    P[0][1]     -= K[0] * P[0][1];
    P[1][0]     -= K[1] * P[0][0];
    P[1][1]     -= K[1] * P[0][1];
    dist=dist_p;
    vel=vel_p;
    return dist;
}
void kalman::setdist(double newdist) { dist = newdist; };
void kalman::setQdist(double newQ_dist) { Q_dist = newQ_dist; };
void kalman::setQvelBias(double newQ_velBias) { Q_velBias = newQ_velBias; };
void kalman::setRdist(double newR_dist) { R_dist = newR_dist; };

double kalman::getVel(void) { return vel; };
double kalman::getQdist(void) { return Q_dist; };
double kalman::getQbias(void) { return Q_velBias; };
double kalman::getRdist(void) { return R_dist; };
