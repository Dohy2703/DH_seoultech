#include "CServo.h"

CServo cservo;

void setup() {
  Serial.begin(9600);
  cservo.attach(3);
  cservo.setAngleLimit(0, 90);
  cservo.setInitAngle(0, true);  // init angle, inc(+:1, -:0)
  cservo.setGrabPeriod(1000);  // set between 0~65535, recommended : 1000
  cservo.attachFSR(A0); 
}

void loop() {
  String user_input;
   
  while(Serial.available())
  {
    user_input = Serial.readStringUntil('\n'); // Get Input from Serial Monitor
    if(user_input =="1"){
      cservo.enable_grab();
    }
  
    else if(user_input =="2"){
      cservo.grabReset();
    }
  }

  cservo.grab(90);

  cservo.checkFSR();
  cservo.printFSR();
}
