#ifndef Capstone_Servo_H_
#define Capstone_Servo_H_

#include <stdint.h>
#include <Arduino.h>
#include <Servo.h>

/* Parameters */
//#define servoPin 1 // assign later
//#define currentPin A0

class CServo : public Servo
/* Servo method : attach(), write(), writeMicroseconds(), 
                  read(), attached(), detach() */
{ 
  private:
    uint8_t fsrPin;
    uint8_t fsrRead; // limited in 0~1000
    
    uint8_t min_angle, max_angle;
    uint8_t prev_angle, init_angle; // limited to 0~90 in attach method
    
    bool inc = true;
    uint8_t inc_angle = 1;

    bool grab_ON = false;
    bool grab_flag1, grab_flag2;
    uint8_t grab_mode; /* 0:no grab. 1:grab, 2: stop */
    uint16_t ang_diff_period;
    unsigned long prev_millis, prev_fsr_millis;
    
  public:
    CServo() { 
      init_angle = 0; 
      min_angle = 0;
      max_angle = 90;  
      ang_diff_period = 10;
      grab_mode = 0;
    }
    
    bool check = false;

    uint8_t angle;
    
    /* FSR sensor */
    void attachFSR(uint8_t fsrPin_);
    void checkFSR();
    void printFSR();

    /* servo motor */
    void setAngleLimit(uint8_t min_angle_, uint8_t max_angle_);
    void setInitAngle(uint8_t init_angle_, bool inc_ = true);
    void setGrabPeriod(uint16_t period_ = 90);
    void enable_grab();
    void grab(uint8_t target = 90);
    void grabReset();
};



#endif
