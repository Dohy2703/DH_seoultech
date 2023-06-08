#include "mbed.h"
#include "Matrix.h"


Matrix ::Matrix(){

}
void Matrix::multiply3x3_3x3(float result[][3], float a[][3], float b[][3]) {

  int ROWS=3;
  int COLS=3;
  int num=3;
    for (int i=0; i < ROWS; i++) {
       for (int j=0; j < COLS; j++) {
        float sum=0;
          for (int k=0; k < num; k++) {
            sum+=a[i][k]*b[k][j];
          }
          result[i][j]=sum;
       }
    }
  }


  void Matrix::multiply3x2_2x2(float result[3][2], float a[3][2], float b[2][2]) {

  int ROWS=3;
  int COLS=2;
  int num=2;

      for (int i=0; i < ROWS; i++) {
       for (int j=0; j < COLS; j++) {
        float sum=0;
          for (int k=0; k < num; k++) {
            sum+=a[i][k]*b[k][j];
          }
          result[i][j]=sum;
       }
    }
  }

  void Matrix::multiply3x3_3x2(float result[3][2], float a[3][3], float b[3][2]) {

  int ROWS=3;
  int COLS=2;
  int num=3;

      for (int i=0; i < ROWS; i++) {
       for (int j=0; j < COLS; j++) {
        float sum=0;
          for (int k=0; k < num; k++) {
            sum+=a[i][k]*b[k][j];
          }
          result[i][j]=sum;
       }
    }

  }

  void Matrix::multiply3x2_2x3(float result[3][3], float a[3][2], float b[2][3]) {

  int ROWS=3;
  int COLS=3;
  int num=2;

      for (int i=0; i < ROWS; i++) {
       for (int j=0; j < COLS; j++) {
        float sum=0;

          for (int k=0; k < num; k++) {
            sum+=a[i][k]*b[k][j];
          }
          result[i][j]=sum;
       }
    }

  }


  void Matrix::multiplyVec3x3(float result[], float a[][3], float b[]) {

  int ROWS=3;
  int COLS=3;

    for (int i=0; i < ROWS; i++) {
      float sum=0;
       for (int j=0; j < COLS; j++)  {
        sum+=a[i][j]*b[j];
          }
          result[i]=sum;
       }
  }


  void Matrix::multiplyVec2x3(float result[], float a[2][3], float b[]) {

  int ROWS=2;
  int COLS=3;
    for (int i=0; i < ROWS; i++) {
      float sum=0;
       for (int j=0; j < COLS; j++)  {
        sum+=a[i][j]*b[j];
          }
          result[i]=sum;
       }
  }

  void Matrix::multiplyVec3x2(float result[], float a[3][2], float b[]) {

  int ROWS=3;
  int COLS=2;
    for (int i=0; i < ROWS; i++) {
      float sum=0;
       for (int j=0; j < COLS; j++)  {
        sum+=a[i][j]*b[j];
          }
          result[i]=sum;
       }
  }

  void Matrix::addMat3x3(float result[][3], float a[][3], float b[][3]){

    int ROWS=3;
    int COLS=3;

    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLS;j++){
        result[i][j] = a[i][j]+b[i][j];

      }
    }
  }

  void Matrix::subMat3x3(float result[][3], float a[][3], float b[][3]){

    int ROWS=3;
    int COLS=3;
    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLS;j++){
        result[i][j] = a[i][j]-b[i][j];
      }
    }
  }

  void Matrix::subVec2x1(float result[], float a[], float b[]){

    int ROWS=2;

    for(int i=0;i<ROWS;i++){
        result[i] = a[i]-b[i];
    }
  }

  void Matrix::subVec3x1(float result[], float a[], float b[]){

       int ROWS=3;

       for(int i=0;i<ROWS;i++){
           result[i] = a[i]-b[i];
       }
     }


  void Matrix::addVec3x1(float result[], float a[], float b[]){

    int ROWS=3;

    for(int i=0;i<ROWS;i++){
        result[i] = a[i]+b[i];
    }
  }
  void Matrix::transpose3x3(float result[][3], float a[][3]){

    int ROWS=3;
    int COLS=3;
    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLS;j++){
        result[i][j] = a[j][i];
      }
    }
  }
  void Matrix::transpose3x2(float result[2][3], float a[3][2]){

    int ROWS=2;
    int COLS=3;
    for(int i=0;i<ROWS;i++){
      for(int j=0;j<COLS;j++){
        result[i][j] = a[j][i];
      }
    }
  }
  void Matrix::inv2x2(float result[2][2], float a[2][2]){
      float det=a[0][0]*a[1][1]-a[0][1]*a[1][0];
      result[0][0]=a[1][1]/det;
      result[0][1]=-a[0][1]/det;
      result[1][0]=-a[1][0]/det;
      result[1][1]=a[0][0]/det;
  }
