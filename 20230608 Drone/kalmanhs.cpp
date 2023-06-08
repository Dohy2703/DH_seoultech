#include "kalmanhs.h"

#include "mbed.h"
#define SAMPLETIME 0.01f
Serial pchs(USBTX, USBRX);
Matrix matrix;
kalmanhs::kalmanhs(){
    


    A[0][0]=1.0f;   A[0][1]=SAMPLETIME;     A[0][2]=SAMPLETIME*SAMPLETIME/2;
    A[1][0]=0;      A[1][1]=1.0f;   A[1][2]=SAMPLETIME;
    A[2][0]=0;      A[2][1]=0;      A[2][2]=1.0f;


    A_t[0][0]=1.0f;                         A_t[0][1]=0;                 A_t[0][2]=0;
    A_t[1][0]=SAMPLETIME;                  A_t[1][1]=1.0f;               A_t[1][2]=0;
    A_t[2][0]=SAMPLETIME*SAMPLETIME/2;     A_t[2][1]=SAMPLETIME;         A_t[2][2]=1.0f;


}
void kalmanhs::Getinverse(float (*inv)[2]){
    static float det=(P_hat[1][1]+R[0][0])*(P_hat[2][2]+R[1][1])-P_hat[1][2]*P_hat[2][1];

    if(det==0) return ;
    inv[0][0]=(P_hat[2][2]+R[1][1])/det;
    inv[0][1]=(-P_hat[1][2])/det;
    inv[1][0]=(-P_hat[2][1])/det;
    inv[1][1]=(P_hat[1][1]+R[0][0])/det;
}

void kalmanhs::kalmanTask_hs(float newvel, float newacc, float* result){
    
    //pckal.printf("%.3f\n",x[2]);
    kalmmanPredict_hs();
    CalKalmangain();
    kalmanUpdate(newvel,newacc);
    result[0]=SV[0];
    result[1]=SV[1];
    result[2]=SV[2];
}

void kalmanhs::kalmmanPredict_hs(){
    static float tmp[3][3]={{0,0,0},{0,0,0},{0,0,0}};

      matrix.multiplyVec3x3(SV_hat,A,SV);

      matrix.multiply3x3_3x3(tmp,A,P);

      matrix.multiply3x3_3x3(P_hat,tmp,A_t);

      matrix.addMat3x3(P_hat,P_hat,Q);
    
}
void kalmanhs::CalKalmangain(){
  static float inv[2][2]={{0,0},{0,0}};
  static float tmpMat[3][2]={{0,0},{0,0},{0,0}};

  Getinverse(inv);

  matrix.multiply3x3_3x2(tmpMat,P_hat,H_t);///
  matrix.multiply3x2_2x2(K,tmpMat,inv);
    
}
void kalmanhs::kalmanUpdate(float newvel, float newacc){
    static float tmpVec[2]={0,0};
    static float tmpMat[3][3]={{0,0,0},{0,0,0},{0,0,0}};
    float mea[2]={newvel, newacc};
    matrix.multiplyVec2x3(tmpVec,H,SV_hat);
    matrix.subVec2x1(tmpVec,mea,tmpVec);
    matrix. multiplyVec3x2(SV,K,tmpVec);
    matrix.addVec3x1(SV,SV_hat,SV);
    matrix.multiply3x2_2x3(tmpMat,K,H);
    matrix.multiply3x3_3x3(P,tmpMat,P_hat);
    matrix.subMat3x3(P,P_hat,P);
    
}
