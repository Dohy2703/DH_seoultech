#ifndef _Matrix_H
#define _Matrix_H

class Matrix
{
    public :
        Matrix(void);
        void multiply3x3_3x3(float result[][3], float a[][3], float b[][3]);
        void multiply3x2_2x2(float result[3][2], float a[3][2], float b[2][2]);
        void multiply3x3_3x2(float result[3][2], float a[3][3], float b[3][2]);
        void multiply3x2_2x3(float result[3][3], float a[3][2], float b[2][3]);
        void multiplyVec3x3(float result[], float a[][3], float b[]);
        void multiplyVec2x3(float result[], float a[2][3], float b[]);
        void multiplyVec3x2(float result[], float a[3][2], float b[]);
        void addMat3x3(float result[][3], float a[][3], float b[][3]);
        void subMat3x3(float result[][3], float a[][3], float b[][3]);
        void subVec2x1(float result[], float a[], float b[]);
        void subVec3x1(float result[], float a[], float b[]);
        void addVec3x1(float result[], float a[], float b[]);
        void transpose3x3(float result[][3], float a[][3]);
        void transpose3x2(float result[2][3], float a[3][2]);
        void inv2x2(float result[2][2], float a[2][2]);
    private:

};

#endif
