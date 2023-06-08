
double mapping(float x, float b, float c, float d, float e);

double map(double Tx, double Tin_min, double Tin_max, double Tout_min,  double Tout_max);

void Check_CH();

void PPM_Rise();

void PPM_Fall();

void dualPID();

void first_PID(float target_rate, float rate_in, float rate_kp, float rate_ki, float rate_kd, float &rate_pterm, float &rate_iterm, float &rate_dterm, float &output);

void PID_height(float target_height, float height_in, float height_kp, float height_ki, float height_kd, volatile float &height_pterm, volatile float &height_iterm, volatile float &height_dterm, volatile float &output);

void PID_velocity(volatile float& prev_dot_dterm,volatile float &prev_dot, float target_dot, float dot_in, float dot_kp, float dot_ki, float dot_kd, volatile float &dot_pterm, volatile float &dot_iterm, volatile float &dot_dterm, volatile float &dot_output);

void calcdot();

void calcYPRtoDualPID();

void initIR();

void optical_vel();

void set_target_height();

void setup_mpu9250();

void init_sensor();

float constrain(float x, float x_min, float x_max);

void IMU_thread_loop();

void IR_thread_loop();

void print_thread_loop();