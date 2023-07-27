#include "CServo.h"

/* FSR sensor */
void CServo::attachFSR(uint8_t fsrPin_)
{
  fsrPin = fsrPin_;

  pinMode(fsrPin, INPUT);  
}

void CServo::checkFSR()
{
  if (millis() - prev_fsr_millis > 50)
  {
    fsrRead = map(analogRead(fsrPin), 0, 1024, 0, 255);

    /* check FSR sensor detection and stop servo motor */
    if (grab_mode == 1 && fsrRead > 180 && grab_flag1 == true && grab_ON)
    {
      grab_flag1 = false;  // check for first current fluctuation

      grab_mode = 2;
    }
    else if (grab_mode == 1 && fsrRead > 180 && grab_ON)
    {
      grab_flag1 = true; 
    }      
  }
}

void CServo::printFSR()
{
  Serial.println(fsrRead);
}


/* servo motor */
void CServo::setAngleLimit(uint8_t min_angle_, uint8_t max_angle_)
{
  min_angle = min_angle_;
  max_angle = max_angle_;
}

void CServo::setInitAngle(uint8_t init_angle_, bool inc_ = true)
{
  /* you should use CServo::attach() and CServo::setAngleLimit()
     before using this member function */
  inc = inc_;  // increase or decrease angle when grab
  init_angle = constrain(init_angle_, min_angle, max_angle);
  this->write(init_angle);
}

void CServo::setGrabPeriod(uint16_t period_ = 90)
{
  /* you should use CServo::setAngleLimit()
     before using this member function */
//  assert(period_ != 0);  // there is no <cassert> in arduino...
  if (period_ == 0) ang_diff_period = 1000;
  else  
  ang_diff_period = period_;
}

void CServo::enable_grab()
{
  grab_ON = true;  
}


void CServo::grab(uint8_t target = 90)
{
  if (grab_ON == false)
  {
    return;
  }

  unsigned long time_gap = millis() - prev_millis;
  int inc_sign = inc?1:-1;

  switch (grab_mode)
  {
    case 0:   // set init time and ready for grab
      grab_mode = 1;
      this->write(init_angle);
      prev_millis = millis();
      break;
      
    case 1:   // grab 
      inc_angle = constrain(int( (max_angle - min_angle) * time_gap / ang_diff_period ), 0, 180);
      angle = constrain(init_angle + inc_angle*inc_sign, min_angle, max_angle);  

      this->write(angle);
  
      prev_angle = angle;

      if ( (inc == true && angle >= target) || (inc == false && angle <= target) )
      {
        grab_mode = 3;
        prev_millis = millis();
      }
      break;

    case 2:   // succeeded to grab something
      this->write(prev_angle);

      grab_ON = false;
      break;

    case 3:   // failed to catch anything
      inc_angle = constrain(int( (max_angle - min_angle) * time_gap / ang_diff_period ), 0, 180);
      angle = constrain(target - inc_angle*inc_sign, min_angle, max_angle);
     
      this->write(angle);

      prev_angle = angle;

      if ( (inc == true && angle <= init_angle ) || (inc == false && angle >= init_angle) )
      {
        grab_mode = 0;
        grab_ON = false;
        prev_millis = millis();
      }
      
      break;

//    case 4:  // reset and go back to original angle
//      if (angle==init_angle){
//        grab_mode = 0; 
//        
//      }
  }
}

void CServo::grabReset(){
  grab_mode = 0;
  grab_ON = false;
  this->write(init_angle);
  grab_flag1 = false;
}
