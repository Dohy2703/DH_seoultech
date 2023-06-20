#include "CEncoder.h"

#define ENC_L_4 4
#define MOTOR_RATIO 49.

CEncoder* CEncoder::instances[2] = {NULL, NULL};

unsigned long cnt_;

//------------variable(upper)------------//
uint8_t motor_left_pwm=0;
uint8_t motor_right_pwm=0;
bool motor_left_dir = 0;    //front: 1, back: 0
bool motor_right_dir = 0;   //front: 1, back: 0

uint8_t pwm_L = 0;
double pwm_R = 0;
//------------variable(lower)------------//

typedef struct _Motor{
  uint8_t pwm = 0;
  bool dir = 0;       //front: 1, back: 0
} Motor;

//------------struct(upper)------------//
Motor motor_right;
Motor motor_left;

#define ENC_L_2 2  // left
#define ENC_L_4 4
#define ENC_R_7 7  // right
#define ENC_R_8 8

CEncoder CMotor_L;
CEncoder CMotor_R;

#define LED_PIN 13
#define motorDirL 12
#define motorDirR 11
#define motorPwmL 6
#define motorPwmR 9
#define ENC_L_2 2  // left
#define ENC_L_4 4
#define ENC_R_7 7  // right
#define ENC_R_8 8


//------------struct(lower)------------//

//long tim1=0;
//long tim2=0;


void setup() {
  // put your setup code here, to run once:
//  setPinMode();
  pinMode(motorDirL, OUTPUT);
  pinMode(motorDirR, OUTPUT);
  pinMode(motorPwmL, OUTPUT);
  pinMode(motorPwmR, OUTPUT);
  Serial.begin(9600);

//  tim1 = millis();
  CMotor_L.begin(ENC_L_2, ENC_L_4, true);
  CMotor_R.begin(ENC_R_7, ENC_R_8, false);
}

void loop() {
  // put your main code here, to run repeatedly:
//  currentMillis = millis();

  String user_input;
   
  while(Serial.available())
  {
    user_input = Serial.readStringUntil('\n'); // Get Input from Serial Monitor
    if(user_input =="1"){
      pwm_L = 25;
      pwm_R = 0.2;
    }
    else if(user_input =="2"){
      pwm_L = 50;
      pwm_R = 0.4;
    }
    else if(user_input =="3"){
      pwm_L = 75;
      pwm_R = 0.6;
    }
    else if(user_input =="4"){
      pwm_L = 100;
      pwm_R = 0.8;
    }
    else if(user_input =="5"){
      pwm_L = 125;
      pwm_R = 1.0;
    }
    else if(user_input =="6"){
      pwm_L = 150;
      pwm_R = 1.2;
    }
    else if(user_input =="7"){
      pwm_L = 175;
      pwm_R = 1.4;
    }
    else if(user_input =="8"){
      pwm_L = 200;
      pwm_R = 1.6;
    }
    else if(user_input =="9"){
      pwm_L = 225;
      pwm_R = 1.8;
    }
    else if(user_input =="0"){
      pwm_L = 250;
      pwm_R = 2;
    }
    else{
      Serial.println("Wrong Number");
    }
  }

//  CMotor_R.PrintPulse();
//  CMotor_L.PrintPulse();

  CMotor_R.CountPulse(50, 49);  // interval_time, ratio
//  Serial.println(CMotor_R.pulse);
  CMotor_R.CMDVELtoTarget(pwm_R, 0, 0);
  CMotor_R.EncoderPID();
  CMotor_R.PIDtoPWM();
  analogWrite(motorPwmR, CMotor_R.pwm);
  Serial.print(CMotor_R.current);
  Serial.print('\t');
  Serial.println(CMotor_R.target);

  
//
//  tim2 = millis();
//  // if (tim2-tim1>15000) target_R = 1.41;
//  // else if (tim2-tim1>10000) target_R = 0.90;
//  // else if (tim2-tim1>5000) target_R = 0.39;
//  target_L = 2.3;
//  target_R = 2.3;

//  current_L = abs(ang_vel_left);
//  current_R = abs(ang_vel_right);
//
//  pidControl_to_pwm_L = int((pidControl_L+0.11)*50./0.51);
//  if (pidControl_to_pwm_L <= 15) pidControl_to_pwm_L = 0;
//  motor_left.pwm = constrain(pidControl_to_pwm_L, 0, 255);
//
//  pidControl_to_pwm_R = int((pidControl_R+0.13)*50./0.51);
//  if (pidControl_to_pwm_R <= 15) pidControl_to_pwm_R = 0;
//  motor_right.pwm = constrain(pidControl_to_pwm_R, 0, 255);

  digitalWrite(motorDirL, 0);
  digitalWrite(motorDirR, 0);
  // analogWrite(motorPwmL, motor_left.pwm);
  // analogWrite(motorPwmR, motor_right.pwm);
  analogWrite(motorPwmL, 0);
//  analogWrite(motorPwmR, pwm_R);

}
