#include "turtlebot3_sensor.h"
#define INTERVAL_MS_TO_UPDATE_CONTROL_ITEM 20

static Turtlebot3Sensor sensors;
void update_imu(uint32_t interval_ms);
bool ret; 
extern float gyro_x, gyro_y, gyro_z;
extern float acc_x, acc_y, acc_z;
extern float mag_x, mag_y, mag_z;
extern float ori_x, ori_y, ori_z, ori_w;

void init_imu(){
  ret = sensors.init();
  sensors.initIMU();
  sensors.calibrationGyro();
}

void update_imu(uint32_t interval_ms)
{
  sensors.updateIMU();
  static uint32_t pre_time = 0;
  float* p_imu_data;

  if(millis() - pre_time >= interval_ms){
    pre_time = millis();

    p_imu_data = sensors.getImuAngularVelocity();
    gyro_x = p_imu_data[0];
    gyro_y = p_imu_data[1];
    gyro_z = p_imu_data[2];

    p_imu_data = sensors.getImuLinearAcc();
    acc_x = p_imu_data[0];
    acc_y = p_imu_data[1];
    acc_z = p_imu_data[2];

    p_imu_data = sensors.getImuMagnetic();
    mag_x = p_imu_data[0];
    mag_y = p_imu_data[1];
    mag_z = p_imu_data[2];

    //ori_x, ori_y가 0이 안나와서 일단 0으로 고정시킴. calibration문제이거나 실제로 opencr보드를 좀 기울게 설치해서 그러거나.
    p_imu_data = sensors.getOrientation();
    ori_w = p_imu_data[0];
    ori_x = p_imu_data[1];
    ori_y = p_imu_data[2];
    ori_z = p_imu_data[3];
  }  
}
