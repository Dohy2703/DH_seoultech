#ifndef _kalman_H
#define _kalman_H
// derived from http://blog.tkjelectronics.dk/2012/09/a-practical-approach-to-kalman-filter-and-how-to-implement-it/
class kalman
{
public:
    kalman(void);
    double getdist(double newdist,  double dt);
    
    void setdist(double newdist);
    void setQdist(double newQ_dist);
    void setQvelBias(double newQ_velBias);
    void setRdist(double newR_dist);
    
    double getVel(void);
    double getQdist(void);
    double getQbias(void);
    double getRdist(void);
    

private:
    double P[2][2];         //error covariance matrix
    double K[2];            //kalman gain
    double y;               //dist difference
    double S;               //estimation error

    double vel;            //Vel in deg/s
    double dist;           //kalman dist
    double vel_p;
    double dist_p;
    double bias;            //kalman vel bias

    double Q_dist;         //process noise variance for the dist of the accelerometer
    double Q_velBias;      //process noise variance for the velscope bias
    double R_dist;         //measurement noise variance 
};

#endif
