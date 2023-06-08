#ifndef _kalmanhs_H
#define _kalmanhs_H
#include "mbed.h"
#include "Matrix.h"
// derived from http://blog.tkjelectronics.dk/2012/09/a-practical-approach-to-kalman-filter-and-how-to-implement-it/
class kalmanhs
{
public:
    kalmanhs();
    void kalmanTask_hs(float newvel, float newacc, float* result);
    void kalmmanPredict_hs();
    void CalKalmangain();
    void kalmanUpdate(float newvel, float newacc);
    void Getinverse(float inv[2][2]);

    //void setdist(double newdist);
    //void setQdist(double newQ_dist);
    //void setQvelBias(double newQ_velBias);
    //void setRdist(double newR_dist);
    
    

    

private:
    //float SAMPLETIME=0.01;
    float SV[3]={0,0,0};
    float SV_hat[3]={0,0,0};
    float Q[3][3]={{400,0,0},{0,8000,0},{0,0,1}};    // {400,0,0},{0,8000,0},{0,0,0.6}
    // float R[2][2]={{400000,0},{0,1}};     //{{300000,0},{0,300000000}};반응성이느린거같으면400000 감소 
    float R[2][2]={{400000,0},{0,0.0001}};     //{{300000,0},{0,300000000}};반응성이느린거같으면400000 감소 
    float A[3][3];
    float A_t[3][3];
    float P[3][3]={{1,0,0},
                  {0,1,0},
                  {0,0,1}};         //error covariance matrix
    float P_hat[3][3]={{0, 0, 0},
                      {0, 0, 0},
                       {0, 0, 0}};

    
    
    float H[2][3]={{0,1, 0},
                     {0, 0, 1}};
    float H_t[3][2]={{0,0},
                      {1, 0},
                      {0, 1}};
    float K[3][2]={{0,0},
                  {0,0},
                  {0,0}};            //kalman gain

};
#endif
