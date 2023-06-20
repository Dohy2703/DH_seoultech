#include "CEncoder.h"

void CEncoder::DecodeISR0 ()
{
  if (CEncoder::instances[0] != NULL)
    CEncoder::instances[0]->Decode();
} 

void CEncoder::DecodeISR1 ()
{
  if (CEncoder::instances[1] != NULL)
    CEncoder::instances[1]->Decode();
} 

void CEncoder::begin(uint8_t PinA_, uint8_t PinB_, bool dir_left)
{  
  pinA = PinA_;
  pinB = PinB_;
  left_motor = dir_left;

  pinMode(pinA, INPUT);
  pinMode(pinB, INPUT);

  switch (pinA)
  {
  case 2: 
    attachInterrupt (0, DecodeISR0, CHANGE);
    attachInterrupt (2, DecodeISR0, CHANGE);
    instances [0] = this;
    break;

  case 4:
    attachInterrupt (0, DecodeISR0, CHANGE);
    attachInterrupt (2, DecodeISR0, CHANGE);
    instances [0] = this;
    break;
  
  case 7: 
    attachInterrupt (3, DecodeISR1, CHANGE);
    attachInterrupt (4, DecodeISR1, CHANGE);
    instances [1] = this;
    break;

  case 8: 
    attachInterrupt (3, DecodeISR1, CHANGE);
    attachInterrupt (4, DecodeISR1, CHANGE);
    instances [1] = this;
    break;
  }

  if (left_motor)
  {
    P_GAIN_ = P_GAIN_L;
    I_GAIN_ = I_GAIN_L;
    D_GAIN_ = D_GAIN_L;
  }
  else 
  {
    P_GAIN_ = P_GAIN_R;
    I_GAIN_ = I_GAIN_R;
    D_GAIN_ = D_GAIN_R;
  }
}

void CEncoder::Decode()
{  
  uint8_t pinA_state, pinB_state;

  pinA_state= digitalRead(pinA);
  pinB_state= digitalRead(pinB);
  uint8_t newState=(pinA_state<<1|pinB_state);
  switch((_previousState <<2)|newState){
    case 0b0001: // 0x01 PREV|CURR
    case 0b0111: // 0x07
    case 0b1110: // 0x0E
    case 0b1000: // 0x08
            {pulse_count--; break;}
    case 0b0010: // 0x02
    case 0b1011: // 0x0B
    case 0b1101: // 0x0D
    case 0b0100: // 0x04
            {pulse_count++; break;}
  default:
      break;
  }
  _previousState = newState;
}


void CEncoder::PrintPulse()
{
  Serial.println(pulse_count);
}

void CEncoder::CountPulse(uint16_t interval_ms, uint8_t motor_ratio)
{
  int32_t return_pulse_count;

  if (millis() - prevMillis > interval_ms) { //interval = 20ms
    prevMillis = millis();  // update timer
 
    rpm = 60./((double)interval_ms/1000.) * (double)pulse_count / (13. * 4.) / (double)motor_ratio;  // calculate RPM
    ang_vel = rpm * RPMtoRAD;  // calculate rad/s

    current = abs(ang_vel);  // current rad/s
    
//    s_pulse_R = String(int(pulse_R*1000));   // 나중에 main문에서 s_pulse_R = String(int(CEncoder_R.CountPulse*1000)); 로 받아오기 

    pulse = pulse_count;
   
    pulse_count = 0;  // reset pulse count
  }
}

double CEncoder::CMDVELtoTarget(double lin_x_, double ang_z_, bool *dir)
{
  double angVel_rotate = constrain((ang_z_*WHEEL_SEPARATION/2)/WHEEL_RADIUS,-ANG_SPEED_LIMIT,ANG_SPEED_LIMIT);    
  double dir_term=left_motor==1?-1.:1.;
  double angVel_linear = constrain(lin_x_/WHEEL_RADIUS, -LIN_SPEED_LIMIT, LIN_SPEED_LIMIT);  
  double angVel_merge = angVel_rotate*dir_term + angVel_linear;  // angVel_merge = angular velocity rotate term + angular velocity linear term
  
  *dir=angVel_merge>0?0:1;
  target = abs(angVel_merge);
}

void CEncoder::EncoderPID()
{
  double realError, errorGap;
  double pControl, iControl, dControl;
    
  realError = target - current;             //for P
  accError += realError;                    //for I
  if (current <= 0.07) accError = 0; 

  errorGap = target - current - realError;  //for D  
  pControl = P_GAIN_ * realError;
  iControl = I_GAIN_ * (accError * PID_TIME / 1000);
  dControl = D_GAIN_ * (errorGap / PID_TIME);

  iControl = constrain(iControl, -0.3, 0.3);
  
  dControl = prev_dControl*0.8 + dControl*0.2;
  prev_dControl = dControl;
  pidControl = current + pControl + iControl + dControl;
}

uint8_t CEncoder::PIDtoPWM()
{
  int pidControl_to_pwm;
  
  pidControl_to_pwm = int((pidControl+0.58)*25./1.52);  //PID끈상태
  if (pidControl_to_pwm <= 15) pidControl_to_pwm = 0;
  pwm = constrain(pidControl_to_pwm, 0, 255);
}
