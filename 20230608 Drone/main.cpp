#include "mbed.h"

#include "Thread.h"
#include "rtos.h"
#include <cstdint>
#include "Matrix.h"
#include "MPU9250.h"
#include "VL53L0X.h" //고도 센서
#include "PMW3901.h" //옵티컬
#include "FastPWM.h" //모터 PWM 제어어
#include "PID.h"
#include "drone_main_header.h"
#include "kalmanhs.h"

//----------------------------제어주기-----------------------
#define MCU_CONTROL_RATE 6
#define IR_UPDATE_RATE 12
#define IMU_UPDATE_RATE 6
#define PRINT_RATE 100


// ---------------------------칼만 객체--------------------
kalmanhs kal_hsX;
kalmanhs kal_hsY;


//----------------------------스레드------------------------
Thread IMU_thread(osPriorityRealtime); //osPriorityHigh
Thread PRINT_thread(osPriorityAboveNormal);
Thread IR_thread(osPriorityHigh);
// Thread IMU_thread; //osPriorityHigh
//Thread PRINT_thread;
//Thread IR_thread(osPriorityHigh);


//-----------------------------타이머--------------
int main_flag =0, sub_flag=0;
RawSerial pc(USBTX, USBRX, 115200);
Timer PPM_timer;
Timer remote_timer_P;
Timer remote_timer_Y;
Timer remote_timer_G;
Timer remote_timer_TH;
Timer remote_timer_ROLL;

Timer make_steady_IMU;
Timer ir_i2c_checking;
Timer imu_i2c_checking;


uint64_t lastUpdate = 0, firstUpdate = 0, Now = 0, Work=0;
double dt = 0;           // 한 사이클 동안 걸린 시간 변수 

// ------------------------------motor-----------------------------------------
#define MIN_PULSE_LENGTH 1000
#define MAX_PULSE_LENGTH 2000
FastPWM motorA(PB_5);
FastPWM motorB(PB_4);
FastPWM motorC(PB_10);
FastPWM motorD(PA_8);

int throttleA, throttleB, throttleC, throttleD; // 각 모터에 따른 스로틀 값

//------------------------IR센서--------------------------
bool IR_i2c_timeout = false;
I2C  i2c_(PB_9, PB_8);
VL53L0X  VL53L0X_SENSOR(&i2c_);

const int maxDistance = 2000;
int height; //?????

volatile float target_height = 0;
float height_kp = 0.4;
float height_ki = 0;
float height_kd = 6; //0.01

float height_pterm;
float height_iterm;
float height_dterm;
volatile float height_output;
float pid_prev_height=0;
float prev_height = 0;

char IR_count = 0;
char loop_count = 0;


//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡppm엔코더ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ//
InterruptIn PPM(D2); //인터럽트 신호 D2
int PPM_ch[8]; // PPM엔코더로 입력되는 신호 개수

double ch_array[9];
char PPM_INDEX=0, PPM_init=0;

uint32_t PPM_all[19]; // PPM 파장 2개 만큼 받아온 후, 시작 신호 찾기
uint32_t PPM_signal[8]={0};

char Fmode;

volatile int throttle; //throtle 타겟 값
int Gear;//Gear 값
int prev_ch_array = 1000; // 스로틀 뒬 경우, 최소 값 1000으로 고정정


//-------------MPU----------------------------------------------

MPU9250 mpu9250;
bool imu_steady=false;
bool imu_i2c_timeout = false;

const double RADIAN_TO_DEGREE = 180 / PI;  
const double ALPHA = 0.992;

double past_DT = 0;           // 한 사이클 동안 걸린 시간 변수 
double DT = 0;           // 한 사이클 동안 걸린 시간 변수 
double free_dt = 0;


//ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ이중루프 PIDㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
volatile float roll_target_angle = 0;
float roll_angle_in;
float roll_rate_in;
float roll_stabilize_kp = 6;
float roll_stabilize_ki = 3;
float roll_rate_kp = 0.5;
float roll_rate_ki = 0;
float roll_rate_kd = 0;
float roll_stabilize_iterm;
float roll_rate_iterm;
float roll_rate_dterm;
float roll_output;

volatile float pitch_target_angle = 0;
float pitch_angle_in;
float pitch_rate_in;  
float pitch_stabilize_kp = 5;
float pitch_stabilize_ki = 3;
float pitch_rate_kp = 0.4;
float pitch_rate_ki = 0;
float pitch_rate_kd = 0;
float pitch_stabilize_iterm;
float pitch_rate_iterm;
float pitch_rate_dterm;
float pitch_output;
float pitch_output_prev;


volatile float yaw_target_rate = 0.0;
float yaw_rate;
float yaw_rate_in;
float yaw_rate_kp = 6;
float yaw_rate_ki = 0.4;
float yaw_rate_kd = 0;
float yaw_rate_pterm;
float yaw_rate_iterm;
float yaw_rate_dterm;
float yaw_output;


//--------------------------optical flow--------------------
float tmp_x_dot=0, tmp_y_dot=0;
PMW3901 flow(PA_7,PA_6,PA_5,PB_6); //PMW3901(PinName mosi, PinName miso, PinName sclk, PinName csel);

volatile float x_dot, y_dot;
int16_t deltaX, deltaY;

const float theta_px=42.0;
const float theta_py=42.0;
const float Nx=300.0;
const float Ny=300.0;

float prev_x_dot = 0;
float prev_y_dot = 0;

volatile float x_target_dot = 0;
float x_dot_in;
float x_dot_kp = 10;
float x_dot_ki = 0;
float x_dot_kd = 0;
float x_dot_pterm;
float x_dot_iterm;
float x_dot_dterm;
volatile float x_dot_output;

volatile float y_target_dot = 0;
float y_dot_in;
float y_dot_kp = 10;
float y_dot_ki = 0;
float y_dot_kd = 0;
float y_dot_pterm;
float y_dot_iterm;
float y_dot_dterm;
volatile float y_dot_output;

float prev_dot_x=0;
float prev_dot_y=0;
//------------------------------칼만---------------------
float raw_acc_x=0;
float raw_acc_y=0;
float raw_acc_z=0;
float roll_rad = 0 , pitch_rad = 0;

float tmp=0,tmp2 = 0, tmp3 = 0,tmp4= 0;
float tmp_height_pterm = 0,tmp_height_iterm = 0, tmp_height_dterm = 0, tmp_height_output = 0;
float roll_kp=0, roll_ki=0,roll_kd=0,roll_error=0;
volatile float prev_dot_dterm=0;
float prev_dot_dterm_x=0, prev_dot_dterm_y=0;

float kalval_hsX[3]={0,0,0};
float kalval_hsY[3]={0,0,0};


double mapping(float x, float b, float c, float d, float e)
{
  double k = ((d-e)/(b-c));
  double result = k*(x-c)+e;

  return result;
}


double map(double Tx, double Tin_min, double Tin_max, double Tout_min,  double Tout_max)
{
    return (Tx - Tin_min) * (Tout_max - Tout_min) / (Tin_max - Tin_min) + Tout_min;
}


void Check_CH(){
    for(int k=0;k<8;k++){
            if(PPM_ch[k]<700)PPM_ch[k]=700;
            if(PPM_ch[k]>1500)PPM_ch[k]=1500;
    }
    
    for(int i =0;i<8;i++){
            PPM_signal[i]=PPM_ch[i];
    }
    //////////CH1 ROLL /////////////
    // PPM_ch[1] = constrain(PPM_ch[1],700,1500);
    x_target_dot = mapping(PPM_ch[1], 700.0, 1500.0, -15.0, 15.0); // 13 -17 pitch
    //y_target_dot = PPM_ch[1]; // pitch -> ch1

    ////////////CH2 YAW//////////
    yaw_target_rate = mapping(PPM_ch[2], 700.0, 1500.0, -15.0, 15.0); // yaw
    //yaw_target_rate = PPM_ch[2];

    ////////////CH5 Gear////////
    if(PPM_ch[5]<=900) Gear = 0;
    else Gear = 1;

    ///////////CH6 Throttle///////
    PPM_ch[6] = mapping(PPM_ch[6], 700, 1500, 1000, 2000);
    throttle = PPM_ch[6];

    ///////////CH7 PITCH//////////////
    y_target_dot = mapping(PPM_ch[7], 700.0, 1500.0, 15.0, -15.0);
    //x_target_dot = PPM_ch[7]; // roll -> ch7


    // roll_target_angle = y_dot_output;  // 여기 다시 확인
    // pitch_target_angle = x_dot_output;  // 여기 다시 확인

}

void PPM_Rise(){
    PPM_timer.reset();
}

void PPM_Fall(){
    PPM_all[PPM_INDEX] = PPM_timer.read_us();
    PPM_INDEX++;
    if(PPM_INDEX==17){
        for(int i=18;i>-1;i--){
            if(PPM_all[i]>10000){
                PPM_init = i;
            }
        }
        for(int i=0;i<8;i++){
            PPM_ch[i]=PPM_all[PPM_init+i+1]; //
        }
        PPM_INDEX = 0;
        Check_CH();
    }
}


void dualPID(float target_angle,
             float angle_in,
             float rate_in,
             float stabilize_kp,
             float stabilize_ki,
             float rate_kp,
             float rate_ki,
             float rate_kd,
             float &stabilize_iterm,
             float &rate_iterm,
             float &rate_dterm,
             float &output
             ){
  float angle_error;
  float desired_rate;
  float rate_error;
  float stabilize_pterm, rate_pterm;
  float rate_prev_dterm;
  float prev_rate;
  
  angle_error = target_angle - angle_in;

  stabilize_pterm = stabilize_kp * angle_error;

  stabilize_iterm += stabilize_ki * angle_error*dt; //안정화 적분항//
 
  desired_rate = stabilize_pterm;

  rate_error = desired_rate - rate_in;
  
  rate_pterm = rate_kp * rate_error; //각속도 비례항//
  rate_iterm += rate_ki * rate_error * dt; //각속도 적분항//
  rate_dterm = -((rate_in - prev_rate)/dt)*rate_kd;

  rate_dterm = rate_prev_dterm*0.9 + rate_dterm*0.1;

  if(stabilize_iterm > 100)
  {
    stabilize_iterm = 100;
  }
  else if(stabilize_iterm < -100)
  {
    stabilize_iterm = -100;
  }

  if(rate_iterm > 50)
  {
    rate_iterm = 50;
  }
  else if(rate_iterm < -50)
  {
    rate_iterm = -50;
  }


  roll_kp = rate_pterm+rate_iterm;
  roll_ki = stabilize_iterm;
  roll_kd = rate_dterm;

  output = rate_pterm + rate_iterm + stabilize_iterm + rate_dterm;

  prev_rate = rate_in;
  rate_prev_dterm = rate_dterm;
}

void first_PID(float target_rate, float rate_in, float rate_kp, float rate_ki, float rate_kd, float &rate_pterm, float &rate_iterm, float &rate_dterm, float &output)
{
  float rate_error;
  float desired_rate;
  float rate_prev_dterm;
  float prev_rate;

  rate_error = target_rate - rate_in;
  
  rate_pterm = rate_kp * rate_error;
  tmp = rate_pterm;

  rate_iterm += rate_ki * rate_error * dt;
  tmp2 = rate_iterm;
  rate_dterm = -((rate_in - prev_rate)/dt)*rate_kd;
  rate_dterm = rate_prev_dterm*0.9 + rate_dterm*0.1;

  tmp3 = rate_dterm;
  output = rate_pterm + rate_iterm + rate_dterm;

  prev_rate = rate_in;
  rate_prev_dterm = rate_dterm;
}

void PID_height(float target_height, volatile float height_in, float height_kp, float height_ki, float height_kd, volatile float &height_pterm,
 volatile float &height_iterm, volatile float &height_dterm, volatile float &output)
{
  volatile float height_error;
  volatile float height_prev_dterm;
  

  height_error = target_height - height_in;

  height_pterm = height_kp * height_error;
  tmp_height_pterm = height_pterm;
  height_iterm += height_ki * height_error * dt;
  height_dterm = -((height_in - pid_prev_height)/dt)*height_kd;

  //tmp_height_iterm = height_iterm;

  height_dterm = height_prev_dterm*0.9 + height_dterm*0.1;
//   tmp_height_dterm = height_dterm;

  if (height_dterm > 100) {
      height_dterm = 100;
  } 
  else if (height_dterm < -100) {
      height_dterm = -100;
  } 


  output = height_pterm + height_iterm + height_dterm;
  //tmp_height_output = output;
  if(output >= 250)
  {
    output = 250;
  }
  else if (output < -30){
      output = -30;
  }
  
  pid_prev_height = height_in;
  height_prev_dterm = height_dterm;
}

void PID_velocity(volatile float& prev_dot_dterm, volatile float& prev_dot, float target_dot, float dot_in, float dot_kp, float dot_ki, float dot_kd, volatile float &dot_pterm, volatile float &dot_iterm, volatile float &dot_dterm, volatile float &dot_output)
{
  volatile float dot_error;
  

  dot_error = target_dot - dot_in;

  dot_pterm = dot_error * dot_kp;
  dot_iterm += dot_error * dot_ki * dt;
  dot_dterm = -((dot_in - prev_dot)/dt) * dot_kd;

  dot_dterm = prev_dot_dterm*0.9 + dot_dterm*0.1;
  
  dot_output = dot_pterm + dot_iterm + dot_dterm;

  prev_dot_dterm = dot_dterm;
  prev_dot = dot_in;
}

void calcdot()
{
  x_dot_in = kalval_hsX[1];
//   x_dot_in = kalval_hsX;
  PID_velocity(prev_dot_dterm_x, prev_dot_x, x_target_dot, x_dot_in, x_dot_kp, x_dot_ki, x_dot_kd, x_dot_pterm, x_dot_iterm, x_dot_dterm, x_dot_output);

  if(x_dot_output >= 7)
  {
    x_dot_output = 7;
  }
  else if(x_dot_output <= -7)
  {
    x_dot_output = -7;
  }
  
  y_dot_in = kalval_hsY[1];
//   y_dot_in = kalval_hsY;
  PID_velocity(prev_dot_dterm_y, prev_dot_y, y_target_dot, y_dot_in, y_dot_kp, y_dot_ki, y_dot_kd, y_dot_pterm, y_dot_iterm, y_dot_dterm, y_dot_output);

  if(y_dot_output >= 7)
  {
    y_dot_output = 7;
  }
  else if(y_dot_output <= -7)
  {
    y_dot_output = -7;
  }
}

void calcYPRtoDualPID(){
  roll_angle_in = roll;
  roll_rate_in = gx;//angvelFiX

  dualPID(roll_target_angle,
          roll_angle_in,
          roll_rate_in,
          roll_stabilize_kp,
          roll_stabilize_ki,
          roll_rate_kp,
          roll_rate_ki,
          roll_rate_kd,
          roll_stabilize_iterm,
          roll_rate_iterm,
          roll_rate_dterm,
          roll_output
  );

  pitch_angle_in = pitch;
  pitch_rate_in = gy;//angvelFiX

  dualPID(pitch_target_angle,
          pitch_angle_in,
          pitch_rate_in,
          pitch_stabilize_kp,
          pitch_stabilize_ki,
          pitch_rate_kp,
          pitch_rate_ki,
          pitch_rate_kd,
          pitch_stabilize_iterm,
          pitch_rate_iterm,
          pitch_rate_dterm,
          pitch_output
  );

  yaw_rate_in = gz;

  first_PID(yaw_target_rate, yaw_rate_in, yaw_rate_kp, yaw_rate_ki, yaw_rate_kd, yaw_rate_pterm, yaw_rate_iterm, yaw_rate_dterm, yaw_output);
}

void initIR()
{
    VL53L0X_SENSOR.init();
    VL53L0X_SENSOR.setModeContinuous();
    VL53L0X_SENSOR.startContinuous();
}

void optical_vel()
{
  // flow.px -> x좌표 속도, flow.py -> y좌표 속도도
  x_dot = (height*(theta_py*PI/180.0)*flow.py/(dt*Ny) + (height*(gy*PI/180.0))*1.5)/1000.0; //질문 준석이네 코드랑 gx,gy다름
  y_dot = (height*(theta_px*PI/180.0)*flow.px/(dt*Nx) + (height*(gx*PI/180.0))*1.5)/1000.0;
  tmp_x_dot = x_dot;
  tmp_y_dot = y_dot;
  
  x_dot = prev_x_dot*0.9 + x_dot*0.1;
  y_dot = prev_y_dot*0.9 + y_dot*0.1;

  if(x_dot > 5)
  { 
    x_dot = prev_x_dot;
  }
  else if(x_dot < -5)
  {
    x_dot = prev_x_dot;
  }

  if(y_dot > 5)
  {
    y_dot = prev_y_dot;
  }
  else if(y_dot < -5)
  {
    y_dot = prev_y_dot;
  }  

  kal_hsX.kalmanTask_hs(x_dot,raw_acc_x*9.81,kalval_hsX);
  kal_hsY.kalmanTask_hs(y_dot,raw_acc_y*9.81,kalval_hsY);

  prev_x_dot = x_dot;
  prev_y_dot = y_dot;
}


void set_target_height(){

    if(throttle >= 1300 && throttle < 1700){
        target_height = prev_height;
        loop_count = 0;
    }

    else if(throttle >= 1700){

        loop_count++;
                
        if(loop_count%30 == 0){
            target_height += 5;
        }
    }
            
    else if(throttle < 1300){

        loop_count++;
                
        if(loop_count%30 == 0){
            target_height -= 5;
        }
    }

    target_height = constrain(target_height,0,500);

    prev_height = height;
}

void setup_mpu9250(){
    uint8_t whoami = mpu9250.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);  // Read WHO_AM_I register for MPU-9250
    pc.printf("I AM 0x%x\t", whoami); pc.printf("I SHOULD BE 0x71\n\r");
    mpu9250.resetMPU9250(); // Reset registers to default in preparation for device calibration
    mpu9250.MPU9250SelfTest(SelfTest); // Start by performing self test and reporting values 
    
    mpu9250.calibrateMPU9250(gyroBias, accelBias); // Calibrate gyro and accelerometers, load biases in bias registers  
    pc.printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",gyroBias[0],gyroBias[1],gyroBias[2],accelBias[0],accelBias[1],accelBias[2]);
    mpu9250.initMPU9250();
    mpu9250.initAK8963(magCalibration);
    pc.printf("%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",gyroBias[0],gyroBias[1],gyroBias[2],accelBias[0],accelBias[1],accelBias[2]);
    mpu9250.getGres();
    mpu9250.getAres();
    mpu9250.getMres();

    // pc.printf("magcal start\n");
    // mpu9250.magcalMPU9250(magbias, magScale);
    // pc.printf("%.3f %.3f %.3f // %.3f %.3f %.3f\n",magbias[0],magbias[1],magbias[2],magScale[0],magScale[1],magScale[2]);
    // wait(100);
}

void init_sensor(){
    setup_mpu9250(); // MPU 셋업
    initIR(); // IR센서 셋업
    flow.init(); // 옵티컬 셋업업
}

float constrain(float x, float x_min, float x_max){
    return (x>x_max)? x_max: ((x<x_min)? x_min:x);
}

void IMU_thread_loop(){
    
    uint64_t Now_M,Work_M;
    float axx = 0, ayy = 0, azz = 0;
    float last_yaw = 0;
    char a = 123;

    pc.printf("imu_start\n");

    bool warning1_imu = true,warning2_imu = true;

    
    make_steady_IMU.start();

    while(1){
        Now_M=rtos::Kernel::get_ms_count();
        mpu9250.get_data();
        if(make_steady_IMU.read_ms()>=4000) {
            imu_steady = true;
            pc.printf("steady\n\n\n");
            make_steady_IMU.stop();
            break;
        }
        Work_M=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(IMU_UPDATE_RATE-(Work_M-Now_M)));
    }    
    imu_i2c_checking.start();

    while(1){
        imu_i2c_checking.reset();
        Now_M=rtos::Kernel::get_ms_count();


        int chk1_im = imu_i2c_checking.read_ms(); // IMU i2c 응답 시간 측정해서, 50ms 이상 2번 확인되면 통신 중단
        if (warning1_imu || warning2_imu) {
            //pc.printf("read_imu\n");
            mpu9250.get_data(); //IMU 값 업데이트 (마호니 업데이트-지자기 O)
            // yaw_rate = (yaw-last_yaw)/0.006;
            // last_yaw = yaw; 
        }
        else imu_i2c_timeout = true;
        
        int chk2_im = imu_i2c_checking.read_ms();

        if(chk2_im-chk1_im>=50) {
            if (warning1_imu ==true) warning1_imu = false;
            else warning2_imu = false;
        }
    
        roll_rad=roll*PI/180; //라디안 변환
        pitch_rad=pitch*PI/180;
        axx = ax+sin(pitch_rad);
        ayy = ay-sin(roll_rad);
        azz = az-cos(pitch_rad)*cos(roll_rad);

        //X,Y,Z(절대좌표계 기준) 가속도 구함 (칼만용)
        raw_acc_y= -(axx*cos(pitch_rad)+(azz)*sin(pitch_rad)); 
        raw_acc_x= (axx*sin(roll_rad)*sin(pitch_rad)+ayy*cos(roll_rad)-(azz)*sin(roll_rad)*cos(pitch_rad));
        raw_acc_z= -axx*cos(roll_rad)*sin(pitch_rad) + ayy*sin(roll_rad)+(azz)*cos(roll_rad)*cos(pitch_rad);

        Work_M=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(IMU_UPDATE_RATE-(Work_M-Now_M)));
    }
}

void IR_thread_loop(){

    uint64_t Now_IR,Work_IR;

    pc.printf("ir_start\n");

    bool warning1_IR = true, warning2_IR = true;

    while(1){

        Now_IR = rtos::Kernel::get_ms_count();
        ir_i2c_checking.reset();

        int chk1_IR = ir_i2c_checking.read_ms();

        if (warning1_IR || warning2_IR){ // IR 센서 I2C 통신 타임아웃 체크
            //pc.printf("read\n");
            height = VL53L0X_SENSOR.getRangeMillimeters(); // IR 센서 값 읽어오기
        }
        else IR_i2c_timeout = true;

        int chk2_IR = ir_i2c_checking.read_ms();

        if(chk2_IR-chk1_IR>=50) {
            if (warning1_IR ==true) warning1_IR = false;
            else warning2_IR = false;
        }

        Work_IR=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(IR_UPDATE_RATE-(Work_IR-Now_IR)));
    }
}



void print_thread_loop(){
    
    uint64_t Now_p,Work_p;
    
    while(1){
        //pc.printf("flag : %d, %d\n",main_flag,sub_flag);
        Now_p=rtos::Kernel::get_ms_count();
        //자세제어 확인

        // pc.printf("%.3f, %.3f, %.3f \n", x_dot_in, x_dot, raw_acc_x);
        pc.printf("%.3f, %.3f, %.3f \n", y_dot_in, y_dot, raw_acc_y);
        // pc.printf("%.3f, %.3f, %.3f \n", kalval_hsY[0], kalval_hsY[1], kalval_hsY[2]);



        // 신호기 입력 값 확인인
        // 센서 별 출력 값 확인
        //pc.printf("sen_val : %.3f, %.3f,%d\n",flow.px,flow.py,height);
        Work_p=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(PRINT_RATE-(Now_p-Work_p)));
    }
}

int main()
{
    // ESC calibration : max pwm(2000) 입력 후, 10초 이상 대기, min pwm(1000) 입력
    // motorA.pulsewidth_us(2000);
    // motorB.pulsewidth_us(2000);
    // motorC.pulsewidth_us(2000);
    // motorD.pulsewidth_us(2000);
    // wait_ms(15000);

    // motorA.pulsewidth_us(1000);
    // motorB.pulsewidth_us(1000);
    // motorC.pulsewidth_us(1000);
    // motorD.pulsewidth_us(1000);
    // wait_ms(10000);

    motorA.pulsewidth_us(1000);
    motorB.pulsewidth_us(1000);
    motorC.pulsewidth_us(1000);
    motorD.pulsewidth_us(1000);
    wait_ms(2000);


    init_sensor();

    pc.printf("%p\n",osThreadGetId());

    osThreadSetPriority(osThreadGetId(),osPriorityRealtime7);
    //IMU 스레드 시작 제어주기 6ms, 변경 원하면 mpu9250 헤더에서 deltat 값 변경해야됨


    PPM.enable_irq();
    PPM_timer.start();

    PPM.rise(&PPM_Rise);
    PPM.fall(&PPM_Fall);
    
    bool warning1 = true;
    bool warning2 = true;

    ir_i2c_checking.start();

    IMU_thread.start(&IMU_thread_loop);
    PRINT_thread.start(&print_thread_loop);

    IR_thread.start(&IR_thread_loop);

    //IMU 안정화까지 10초 기다림
    while(1){
        Now=rtos::Kernel::get_ms_count();
        if(imu_steady) break;
        Work=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(MCU_CONTROL_RATE-(Work-Now)));
    }    


    while (true) {

        // main_flag = main_flag+1;
        // if (main_flag>=10000) main_flag = 0;

        Now = rtos::Kernel::get_ms_count();


        flow.read(); // Optical 값 읽어오기 

        optical_vel();

        roll_target_angle = -x_dot_output;  // 여기 다시 확인
        pitch_target_angle = y_dot_output;  // 여기 다시 확인

        if(Gear == 0)
        {
            pitch_output = 0;
            roll_output = 0;
            yaw_output = 0;
            height_output = 0;
            throttleA = 1000;
            throttleB = 1000;
            throttleC = 1000;
            throttleD = 1000;
        }

        else if(Gear == 1)
        {

            dt = (rtos::Kernel::get_ms_count() - Work)/1000.;
                      // 한 사이클 동안 걸린 시간 변수 

            calcdot();

            //dt = (rtos::Kernel::get_ms_count() - Work)/1000.;

            calcYPRtoDualPID();

            set_target_height();

            //dt = (rtos::Kernel::get_ms_count() - Work)/1000.;

            PID_height(target_height, height, height_kp, height_ki, height_kd, height_pterm, height_iterm, height_dterm, height_output);
            
            //YPR + 고도
            throttleA = 1280 - pitch_output - roll_output + yaw_output + height_output;
            throttleB = 1280 - pitch_output + roll_output - yaw_output + height_output;
            throttleC = 1280 + pitch_output - roll_output - yaw_output + height_output;
            throttleD = 1280 + pitch_output + roll_output + yaw_output + height_output;
            
            // // YPR 제어
            // throttleA = -pitch_output - roll_output + yaw_output + throttle;
            // throttleB = -pitch_output + roll_output - yaw_output + throttle;
            // throttleC = pitch_output - roll_output - yaw_output + throttle;
            // throttleD = pitch_output + roll_output + yaw_output + throttle;


            // // pitch pid
            // throttleA = throttle-pitch_output;
            // throttleB = throttle-pitch_output;
            // throttleC = throttle+pitch_output;
            // throttleD = throttle+pitch_output;

            // // roll pid
            // throttleA = throttle - roll_output;
            // throttleB = throttle + roll_output;
            // throttleC = throttle - roll_output;
            // throttleD = throttle + roll_output; 



            // //스로틀 확인용
            // throttleA = throttle;
            // throttleB = throttle;
            // throttleC = throttle;
            // throttleD = throttle;

            throttleA = constrain(throttleA,1000,2000);
            throttleB = constrain(throttleB,1000,2000);
            throttleC = constrain(throttleC,1000,2000);
            throttleD = constrain(throttleD,1000,2000);
        }
        
        if(imu_i2c_timeout || IR_i2c_timeout){
            motorA.pulsewidth_us(1000); //i2c 꺼지면 비행 x
            motorB.pulsewidth_us(1000); //
            motorC.pulsewidth_us(1000); //
            motorD.pulsewidth_us(1000); //
        }

        else{
            motorA.pulsewidth_us(throttleA); 
            motorB.pulsewidth_us(throttleB); 
            motorC.pulsewidth_us(throttleC); 
            motorD.pulsewidth_us(throttleD); 
        }

        prev_height = target_height;  

        //pc.printf("%d\n",Gear);

        Work=rtos::Kernel::get_ms_count();
        ThisThread::sleep_until(rtos::Kernel::get_ms_count()+(MCU_CONTROL_RATE-(Work-Now)));
        //ThisThread::sleep_for((3-(Work-Now)));
    }

}