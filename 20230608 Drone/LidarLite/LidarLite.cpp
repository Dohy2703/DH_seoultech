/*
 *  Library for easy interface of LidarLite with mbed using I2C
 *  
 *  Akash Vibhute   <akash . roboticist [at] gmail . com>
 *  
 *  v0.1, 17/Feb/2015 - First version of library, tested using LPC1768 [powered via mbed 3.3v, no additional pullups on I2C necessary]
 *
 */
 
#include "LidarLite.h"

LidarLite::LidarLite(PinName sda, PinName scl)
{
    i2c_ = new I2C(sda, scl);
    i2c_->frequency(100000); //I2C @ 100kHz
    wait(0.5);
}

int16_t LidarLite::getRange_cm()
{
   return(distance_LL);
}

int16_t LidarLite::getVelocity_cms()
{
    if(velocity_LL < 127)
        return(velocity_LL*10);
    else
        return((velocity_LL-256)*10);
}

void LidarLite::refreshRange()
{
    uint8_t nackack;
    
    char write[2]={SET_CommandReg, AcqMode};
    char read_dist[1]={GET_Distance2BReg};
    char read_vel[1]={GET_VelocityReg};
    
    char dist[2];
    char vel[1];
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, write, 2);
    }
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, read_dist, 1);
    }    
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->read(LIDARLite_ReadAdr, dist, 2);
    }
    distance_LL = ((uint16_t)dist[0] << 8) + (uint16_t)dist[1];
}

void LidarLite::refreshVelocity()
{
    uint8_t nackack;
    
    char write[2]={SET_CommandReg, AcqMode};
    char read_dist[1]={GET_Distance2BReg};
    char read_vel[1]={GET_VelocityReg};
    
    char dist[2];
    char vel[1];
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, write, 2);
    }
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, read_vel, 1);
    }    
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->read(LIDARLite_ReadAdr, vel, 1);
    }
    velocity_LL = (uint16_t)vel[0];
}

void LidarLite::refreshRangeVelocity()
{
    uint8_t nackack;
    
    char write[2]={SET_CommandReg, AcqMode};
    char read_dist[1]={GET_Distance2BReg};
    char read_vel[1]={GET_VelocityReg};
    
    char dist[2];
    char vel[1];
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, write, 2);
    }
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, read_dist, 1);
    }    
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->read(LIDARLite_ReadAdr, dist, 2);
    }
    distance_LL = ((uint16_t)dist[0] << 8) + (uint16_t)dist[1];
    
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->write(LIDARLite_WriteAdr, read_vel, 1);
    }    
    
    nackack=1;
    while(nackack !=0)
    {
        wait_ms(1);
        nackack = i2c_->read(LIDARLite_ReadAdr, vel, 1);
    }
    velocity_LL = (uint16_t)vel[0];
}