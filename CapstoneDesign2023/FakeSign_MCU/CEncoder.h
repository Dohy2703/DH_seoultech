#ifndef Capstone_Encoder_H_
#define Capstone_Encoder_H_

#include <stdint.h>
#include <Arduino.h>

/* Parameters */
#define MOTOR_RATIO 49.
#define RPMtoRAD 0.10471975512
#define PULSEtoRAD 6.283185307/(13*4*49)
#define ENCODER_INTERVAL 20

#define WHEEL_MAX_ANG_VEL 15.4   // maximum angular velocity of wheels
#define WHEEL_RADIUS 0.0525
#define WHEEL_SEPARATION 0.42    // distance from wheel to another wheel
#define MAX_LIN_VEL WHEEL_MAX_ANG_VEL * WHEEL_RADIUS  // maximum linear velocity (x-axis)
#define MAX_ANG_VEL WHEEL_RADIUS * WHEEL_MAX_ANG_VEL / (WHEEL_SEPARATION/2)  // maximum angular velocity (z-axis)
#define LIN_SPEED_LIMIT 15  // set limitation
#define ANG_SPEED_LIMIT 7.6


/* PID GAIN */
#define P_GAIN_L 1.
#define I_GAIN_L 3.
#define D_GAIN_L 0.
#define P_GAIN_R 1.
#define I_GAIN_R 3.
#define D_GAIN_R 0.
#define PID_TIME 10  // PID control time

class CEncoder
{
  
  private:
    uint8_t pinA, pinB;
    uint8_t pinA_state, pinB_state;
    int32_t decodeCNT = 0;
    
    double rpm, ang_vel; 
    double pidControl;
    bool left_motor = true;
    double P_GAIN_, I_GAIN_, D_GAIN_;

    uint32_t prevMillis;
    int32_t prev_pulse_count;
    uint8_t _previousState;
    double prev_dControl = 0;
    double accError = 0;


    static void DecodeISR0 ();
    static void DecodeISR1 ();
        
  public:
    volatile int32_t pulse_count = 0;  // real time encoder pulse
    
    int32_t pulse = 0;  // 우리가 실제로 사용할 펄스
    uint8_t pwm = 0;

    double current, target;
    
    static CEncoder *instances[2];

    /* functions */
    void begin(uint8_t PinA_, uint8_t PinB_, bool dir_left);  // set pins, dir
    void Decode();  // decode pulse count 
    void PrintPulse();  // check pulse
    
    void CountPulse(uint16_t interval_ms, uint8_t motor_ratio);
    double CMDVELtoTarget(double lin_x_, double ang_z_, bool *dir);
    void EncoderPID();
    uint8_t PIDtoPWM();
};


#endif
