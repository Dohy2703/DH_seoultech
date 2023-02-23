#include <Servo.h>

#define cntTime 2000 // Control time. Determine speed of stepping motor(low value -> high speed, micro second, even number)
#define stopPoint 250 // stop point

#define m2stp 6       // stepping motor stp-6 pin
#define m2dir 7       // stepping motor dir-7 pin

Servo myServo;

bool sw = 1;          // if switch On
bool startSignal = false;

bool exitFor = false;

char bin[6] = {'0', '0', '0', '0', '0'};
bool arrived = false;
char drugs[5] = {'t', 's', 'r', 'd'};


unsigned long past;   // timer_previous time
unsigned long now=0;  // timer_current time
int x = 0; // 초기-정지
int list_x[6] = {0, 350, 670, 1040,1440};
unsigned int idx=0;
unsigned long long cnt=0; // cnt++ for each loop 

char rasp = '0';
byte mode = 0;

float IR;
float prevIR;
void swON(){
   x = 0;              // if switch ON, initialize x
//  startSignal = true;  // when started,
}


void setup() {
  //attachInterrupt(2,swOFF,FALLING);
  
  Serial.begin(115200);           // Not using when connected to PI
//  pinMode(2, INPUT_PULLUP);     // INPUT PULLUP to prevent floating of switch
//  attachInterrupt(0,swON,RISING);
  pinMode(m2stp, OUTPUT);       //m2stp stp 
  pinMode(m2dir, OUTPUT);       //m2dir dir 
  pinMode(9, INPUT);
  myServo.attach(11);
  x=0;
}

void loop() {
  past = now;     // To fix control time
  cnt++;          // increase counter
    IR=digitalRead(9);

    if (IR == 1) Serial.print("y\n");
    else Serial.println("x");

//    Serial.println(IR);
  while (Serial.available()){
//    idx = Serial.parseInt();
    rasp = Serial.read();
    Serial.flush();
  }

  
//  Serial.println(a);
//  
  if (rasp == '0') {
    mode = 0;
    //myServo.write(0);
  }
  else if (rasp == 'R') {
    mode = 1;
    //myServo.write(75);
//    rasp = '0';
  }
  else if (rasp == 'f'){
    mode = 2;
//    rasp = '0';
  }
  else if (rasp == 'a'){
    mode = 3;
//    rasp = '0';
  }

  
//ㅡㅡㅡㅡㅡㅡㅡㅡㅡMode 0:Wait for startㅡㅡㅡㅡㅡㅡㅡㅡ//
if (mode == 0){  
  move_x(0);

  if (x==0) myServo.write(75);
}

//ㅡㅡㅡㅡㅡㅡㅡㅡㅡMode 1:Drugs Recognitionㅡㅡㅡㅡㅡㅡㅡㅡ//
if (mode == 1){  
  if (rasp == 't' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='0') {
        idx = i;
        bin[i]='t';
        exitFor = true;
        arrived = false;
        break;
      }
    }
  }
  else if (rasp == 's' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='0') {
        idx = i;
        bin[i]='s';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  else if (rasp == 'r' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='0') {
        idx = i;
        bin[i]='r';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  else if (rasp == 'd' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='0') {
        idx = i;
        bin[i]='d';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  move_x(list_x[idx]);

  if (arrived) {
    delay(1000);
//    myServo.write(40);
//    delay(500);
//    myServo.write(35);
//    delay(100);
    myServo.write(30);
    delay(100);
    myServo.write(25);
    delay(100);
    myServo.write(20);
    delay(100);
    myServo.write(15);
    delay(100);
    myServo.write(10);
    delay(100);
    myServo.write(5);
    delay(100);
    myServo.write(0);
    delay(1000);
    arrived = false;
    rasp = '0';
    exitFor = false;
    idx = 0;
    mode = 0;
  }
}

//ㅡㅡㅡㅡㅡㅡㅡㅡㅡMode 2:Drug Recognitionㅡㅡㅡㅡㅡㅡㅡㅡ//
if (mode == 2){
  myServo.write(0);  // 수정
  if (rasp == 't' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='t') {
        idx = i;
        bin[i]='0';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  else if (rasp == 's' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='s') {
        idx = i;
        bin[i]='0';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  else if (rasp == 'r' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='r') {
        idx = i;
        bin[i]='0';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  else if (rasp == 'd' && !exitFor) {
    for(byte i=1; i<5; i++){
      if (bin[i]=='d') {
        idx = i;
        bin[i]='0';
        arrived = false;
        exitFor = true;
        break;
      }
    }
  }
  move_x(list_x[idx]);
  if (arrived) {
    myServo.write(75);
    delay(1000);
    arrived = false;
    mode = 0;
    rasp = '0';
    exitFor = false;
    idx = 0;
  }  
}

//ㅡㅡㅡㅡㅡㅡㅡㅡㅡMode3:send out the drugㅡㅡㅡㅡㅡㅡㅡㅡ//
if (mode == 3){
  myServo.write(0);
  delay(1000);
  myServo.write(75);
  delay(1000);
  mode = 0;
  rasp = '0';
}

  prevIR = IR;

//  Serial.print(x);
//  Serial.print('\t');
//  Serial.print(rasp);
//  Serial.print('\t');
//  Serial.print(mode);
//  Serial.print('\t');
//  Serial.print(idx);
//  Serial.print('\t');
//  Serial.print(bin[1]);
//  Serial.print('\t');
//  Serial.print(bin[2]);
//  Serial.print('\t');
//  Serial.print(bin[3]);
//  Serial.print('\t');
//  Serial.print(bin[4]);
//  Serial.print('\t');
//  now = micros();
//  Serial.println(now-past);   
  while ( (now - past) <= cntTime ) now = micros();    // Fix control time - to control stepping motor
}

void move_x(int x_target){
    if (x<x_target) {
      digitalWrite(m2dir,HIGH);
      x++;
      if (cnt%2==0) digitalWrite(m2stp,HIGH);
      else digitalWrite(m2stp,LOW); 
    }
    else if (x>x_target) {
      digitalWrite(m2dir,LOW);
      x--;
      if (cnt%2==0) digitalWrite(m2stp,HIGH);
      else digitalWrite(m2stp,LOW);
    }
    else if (x!=0 && x==x_target){
      digitalWrite(m2stp,LOW);
      arrived = true;
    }
    else {
      digitalWrite(m2stp,LOW);
    }
  
}
