#include "ServoTimer2.h" // Servo.h가 TimerOne.h를 참조하므로 두 라이브러리를 함께 사용 불가. 따라서 Servotimer2.h 사용 이 때 angle 750(0도)~2250(180도)임에 주의
#include <TimerOne.h>
#include "LiquidCrystal_I2C.h"

#define trig 12 // 초음파 trig
#define echo 2 // 초음파 echo를 인터럽트 핀0으로 사용
#define DAngle 28 // unit angle = 180/DAngle이 되는 값. 실제로는 140도라서 140/DAngle=5가 되는 값이 28로 잡음
#define TimerInterval 50000 // 타이머 시간초 간격 -> 서보모터 회전속도(작을수록 빠름) us단위. 50000 권장
#define balsaSW 3

//객체 생성
ServoTimer2 Servo1; // Servo1 객체 생성(포신 왼쪽)
ServoTimer2 Servo2; // Servo2 객체 생성(포신 오른쪽)
ServoTimer2 CenterServo; //포신 좌우 조절 서보 객체 생성
ServoTimer2 balsaServo; //발사를 조절하는 서보 객체 생성 => continuous servo FS90R 사용
//LiquidCrystal_I2C lcd(0x27,20,4);  // 접근주소: 0x3F or 0x27

//x,y 계산 관련 변수
int x,y; //서보모터 1이 계산한 x,y
int x2,y2;
float savedx, savedy;       //발사버튼 눌렸을 때 저장된 x,y값

//발사 관련 변수
int make2Second=0;        //BALSSA 함수에서 약 2초 세고 발사
int MODE=1;               //MODE1:자동인식, MODE2:발사모드

//함수원형
void getDistance();       //거리 측정
void countXY();           //서보1 측정된 거리로 xy값 반환
void countXY2();          //서보2 측정된 거리로 xy값 반환
void countRXY();          //xy값을 통해 중앙서보모터 각도 계산
void ActiveCode();        //MODE1 코드 실행(적 탐지)
void BALSSA();            //MODE2 코드 실행(발사)

//그 외 변수
volatile bool state=false;          // 타이머 tick마다 신호(state) 보내줌
int cntServo=0;                     //루프 안에서의 서보카운트
int cntServo2=0;                    //루프 안에서의 서보카운트2
volatile bool ServoEnable=true;     //서보 동작시킬 것인가
volatile bool ServoEnable2=true;
volatile int distance;              //초음파 센서가 읽은 거리
volatile int distance2;             //psd 센서가 읽은 거리
int mappedAngle=750;                // 0~180도 범위를 ServoTimer2 범위(750~2250)로 맵핑
int mappedAngle2=2250;              // 서보2
float servo1Angle=20;               //서보모터1의 초기 각도값(countXY)에서 사용
float servo2Angle=20;               //서보모터2의 초기 각도값(countXY2)에서 사용
bool wave_finished = true;          //pulseIn 딜레이 없애기 - github DooHaKim/Radar_non_block.c
unsigned long _start, _end;
float theta=70;                       //CenterServo 각도를 위해 x,y 아크탄젠트를 통해 세타값 계산하기 위한 변수. 초기값 70
int mappedCSA=0;                    //맵핑된 Center Servo Angle
bool chkCSA = false;                //Center Servo Angle를 받아왔는가?


void Out_Wave(){   //초음파 딜레이 없애기 함수1-1
  wave_finished = false;
  digitalWrite(trig,HIGH);
  delayMicroseconds(10);
  digitalWrite(trig,LOW);
}

void ISR_echo(){      //초음파 딜레이 없애기 함수1-2
  switch(digitalRead(echo)){
    case HIGH:
      _start = micros();
      break;
    case LOW:
      _end = micros();
      wave_finished = true;
      break;
    }
  }


void tick(void){      //타이머 인터럽트 tick
  state=true;         //시간간격마다 cntServo 변수값 1씩 증가시키는 변수
  getDistance();      
  countXY();         
  countXY2();

  //Serial.print("x= ");Serial.print(x);Serial.print(", y= ");Serial.print(y);Serial.print(", x2= ");Serial.print(x2);Serial.print(", y2= ");Serial.println(y2);
  //Serial.print("distance1= ");Serial.print(distance);Serial.print(", distance2= ");Serial.println(distance2);
}

void MODE_C2(){     //인터럽트 스위치 눌리면 모드2 시작(BALSSA 함수 실행)
  savedx=(x+x2)/2;
  savedy=(y+y2)/2;
  MODE=2;
}

//초기 세팅  
void setup() {
  pinMode(balsaSW, INPUT_PULLUP);   //인터럽트 발사 스위치(3핀) 내부풀업
  pinMode(4,INPUT_PULLUP);          //스위치(4핀) 내부풀업
  Serial.begin(115200);             //시리얼 모니터 통신속도
  pinMode(trig,OUTPUT);             //초음파 trig
  pinMode(echo,INPUT);              //초음파 echo
  attachInterrupt(0, ISR_echo, CHANGE);               //초음파 인터럽트0
  attachInterrupt(1, MODE_C2, FALLING);                //발사
  Servo1.attach(10);                //서보 9,10번핀 사용->아날로그 신호(PWM)
  Servo2.attach(9);                 //서보
  CenterServo.attach(11);           //포신 좌우조절 서보 11번 핀(PWM)
//  balsaServo.attach(5);             //발사를 조절하는 서보
  Timer1.initialize(TimerInterval); //시간초 간격 500000->0.5s
  Timer1.attachInterrupt(tick);     //시간초 간격마다 tick의 동작
  Out_Wave();                       //초음파 딜레이 없애기 함수1
//  lcd.init();                       //lcd 시작
//  lcd.backlight();                  //lcd 배경불빛
  Servo1.write(750);                //서보 초기 각도
  Servo2.write(2250);               
//  lcd.setCursor(0,0);               //lcd 초기 입력값
//  lcd.print("Press Start");
//  lcd.setCursor(0,1);
//  lcd.print("Loading...");
  CenterServo.write(1500);
  int sw=digitalRead(4);            //시작버튼 핀4 지정
  balsaServo.write(1500);
  while(sw==1){                     //시작 버튼 눌릴 때까지 대기
    sw=digitalRead(4);
  }
//  lcd.clear();
  getDistance();                    //처음 거리 계산
}


//메인 루프
void loop() {  
    if (MODE==1) ActiveCode();
    if (MODE==2) BALSSA();

}



//모드1 함수
void ActiveCode(){
    if (state){ //시간간격마다 cntServo 올려줌
      if (ServoEnable) cntServo++;
      if (ServoEnable2) {cntServo2++;
      cntServo2++;}
      state=false;
    }
    
    if (cntServo==2*DAngle) cntServo=0;
    if (cntServo2==2*DAngle) cntServo2=0;
  
    if (distance < 30) ServoEnable=false;
    else ServoEnable=true;

    if (distance2 < 30) ServoEnable2=false;     //적외선 센서가 너무 사물 잘 인식해서 거리 줄임
    else ServoEnable2=true;

    
    
    if (cntServo<DAngle) mappedAngle=map(cntServo*(140/DAngle),0,140,750,2250); //ServoTimer2 사용했으므로 맵핑
    else if (cntServo<2*DAngle) mappedAngle=map(280-cntServo*(140/DAngle),0,140,750,2250);

    if (cntServo2<DAngle) mappedAngle2=map(cntServo2*(140/DAngle),0,140,750,2250);
    else if (cntServo2<2*DAngle) mappedAngle2=map(280-cntServo2*(140/DAngle),0,140,750,2250); //ServoTimer2 사용했으므로 맵핑


    if (ServoEnable==true){
      Servo1.attach(10);
      Servo1.write(mappedAngle);
    }
    else {
      Servo1.detach();                  //detach 쓰지 않으면 서보모터가 틱틱거림
//      lcd.setCursor(0,0);
//      lcd.print("x=");lcd.print(x);lcd.print(", y=");lcd.print(y);lcd.print("  ");
    }
    
    if (ServoEnable2==true){
      Servo2.attach(9);
      Servo2.write(3000-mappedAngle2);  //서보1과 반대 방향 회전
    }
    else {
      Servo2.detach();
//      lcd.setCursor(0,1);
//      lcd.print("x2=");lcd.print(x2);lcd.print(", y2=");lcd.print(y2);lcd.print("  ");
    }
}




//거리계산 함수
void getDistance(){           //센서값으로 거리를 받아서 distance, distance2로 반환
    if(wave_finished)         //초음파 센서값 반환
   {
    distance = (_end - _start) * 0.034 / 2 - 3;
    Out_Wave();
    }

    int volt = map(analogRead(A1),0,1023,0,5000); //적외선 센서값 반환
    distance2 = (23.666/(volt-0.1696)*1000);
    
    if (distance > 300) distance=300;
    if (distance2 > 300) distance=300;
}


//극좌표->xy좌표 좌표변환 함수
void countXY() {
  if (cntServo<DAngle) servo1Angle = 20 + 5*cntServo;
  else if (cntServo<2*DAngle) servo1Angle = 160 - 5*(cntServo-DAngle);
  x=-15+(5+distance)*cos(radians(servo1Angle)); //보정값
  y=distance*sin(radians(servo1Angle))+4;
}

void countXY2() {
  if (cntServo2<DAngle) servo2Angle = 160 - 5*cntServo2;
  else if (cntServo<2*DAngle) servo2Angle = 20 + 5*(cntServo2-DAngle);
  x2=12+distance2*cos(radians(servo2Angle)); 
  if (cntServo>(DAngle/2) && cntServo<(DAngle*3/2)) x2=x2+5;
  y2=distance2*sin(radians(servo2Angle));
  if (y2<4) y2=y2+3;    
}


//모드2 함수(발사)
void BALSSA() { 
  theta=degrees(atan(abs(savedx)/abs(savedy)));
//  theta=atan(abs(savedx)/abs(savedy));
  if (savedx>0) mappedCSA=map(70-theta,0,140,500,2500);  //이론상으로는 70-theta이지만 보정해준 값
  else mappedCSA=map(70+theta,0,140,500,2500);
//  lcd.clear();
//  lcd.setCursor(0,0);
//  lcd.print("Warnimg:Enemy Detected!");
//  lcd.setCursor(0,1);
//  lcd.print("x=");lcd.print(savedx);lcd.print(", y=");lcd.print(savedy);
  MODE=2;
  if (state) make2Second++;
  Serial.print(savedx);Serial.println(savedy);

  Serial.print(theta);Serial.println(mappedCSA);
  
//  if (chkCSA==false) {
    countRXY();
//    chkCSA=true;
    CenterServo.write(mappedCSA);
    delay(500);
//  }
    balsaServo.write(1400);
    delay(2000);
    balsaServo.write(1500);
////    chkCSA=false;
//    lcd.clear();
    MODE=1;
}


//xy->Center Servo 극좌표 역변환
void countRXY(){
  theta=degrees(atan(abs(savedx)/abs(savedy)));
////  theta=atan(abs(savedx)/abs(savedy));
//  if (savedx>0) mappedCSA=map(70-theta,0,140,500,2500);  //이론상으로는 70-theta이지만 보정해준 값
//  else mappedCSA=map(70+theta,0,140,500,2500);
//  lcd.clear();
//  lcd.setCursor(0,0);
//  lcd.print("Warnimg:Enemy Detected!");
//  lcd.setCursor(0,1);
//  lcd.print("x=");lcd.print(savedx);lcd.print(", y=");lcd.print(savedy);
}
