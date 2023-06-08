/** LidarLite mBed I2C driver
*
* @author Akash Vibhute
* @author < akash . roboticist [at] gmail . com >
* @version 0.1
* @date Feb/17/2015 - v0.1 - First version of library, tested using LPC1768 [powered via mbed 3.3v, no additional pullups on I2C necessary]
* @date June/05/2016 - Doc update
*
* @section LICENSE
*
* Copyright (c) 2015 Akash Vibhute
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*/

#ifndef LidarLite_H
#define LidarLite_H

#include <mbed.h>

/* Default I2C Address of LIDAR-Lite. */
#define LIDARLite_WriteAdr  0xc4    /// 8-bit slave write address
#define LIDARLite_ReadAdr   0xc5    /// 8-bit slave read address

/* Commands */
#define SET_CommandReg       0x00   /// Register to write to initiate ranging
#define AcqMode              0x04   /// Value to set in control register to initiate ranging

/* Read Registers */
#define GET_DistanceHBReg    0x0f   /// High byte of distance reading data
#define GET_DistanceLBReg    0x10   /// Low byte of distance reading data
#define GET_Distance2BReg    0x8f   /// Register to get both High and Low bytes of distance reading data in 1 call
#define GET_VelocityReg      0x09   /// Velocity measutement data

/** LidarLite
 *
 *  @section DESCRIPTION
 *  This is the LIDAR Lite, a compact high performance optical distance measurement
 *  sensor from PulsedLight. The LIDAR Lite is ideal when used in drone, robot, or
 *  unmanned vehicle situations where you need a reliable and powerful proximity
 *  sensor but don’t possess a lot of space. All you need to communicate with this
 *  sensor is a standard I2C or PWM interface and the LIDAR Lite, with its range of
 *  up to 40 meters, will be yours to command!
 *
 *  Each LIDAR Lite features an edge emitting, 905nm (75um, 1 watt, 4 mrad, 14mm optic),
 *  single stripe laser transmitter and a surface mount PIN, 3° FOV with 14mm optics receiver.
 *  The LIDAR Lite operates between 4.7 - 5.5VDC with a max of 6V DC and has a current
 *  consumption rate of <100mA at continuous operation. On top of everything else, the
 *  LIDAR Lite has an acquisition time of only 0.02 seconds or less and can be interfaced
 *  via I2C or PWM.
 *
 *  Note: The LIDAR Lite is designated as Class 1 during all procedures of operation,
 *  however operating the sensor without its optics or housing or making modifications to the
 *  housing can result in direct exposure to laser radiation and the risk of permanent eye damage.
 *  Direct eye contact should be avoided and under no circumstances should you ever stare
 *  straight into the emitter.
 *
 *  Features:
 *  Range: 0-40m Laser Emitter
 *  Accuracy: +/- 0.025m
 *  Power: 4.7 - 5.5V DC Nominal, Maximum 6V DC
 *  Current Consumption: <100mA continuous operation
 *  Acquisition Time: < 0.02 sec
 *  Rep Rate: 1-100Hz
 *  Interface: I2C or PWM
 *
 *  Example:
 *  @code
 *  #include "mbed.h"
 *  #include "LidarLite.h"
 *
 *  #define LIDARLite1_SDA p9   //SDA pin on LPC1768
 *  #define LIDARLite1_SCL p10  //SCL pin on LPC1768
 *
 *  LidarLite sensor1(LIDARLite1_SDA, LIDARLite1_SCL); //Define LIDAR Lite sensor 1
 *
 *  Timer dt;
 *
 *  Serial pc(USBTX,USBRX);
 *
 *  int main()
 *  {
 *      pc.baud(921600);
 *      dt.start();
 *
 *      while(1)
 *      {
 *          //sensor1.refreshRange();
 *          //sensor1.refreshVelocity();
 *          sensor1.refreshRangeVelocity();
 *
 *          pc.printf("range: %d cm, velocity: %d cm/s, rate: %.2f Hz\n", sensor1.getRange_cm(), sensor1.getVelocity_cms(), 1/dt.read());
 *          dt.reset();
 *      }
 *
 *  }
 *  @endcode
 */
class LidarLite
{
public:

    /** Creates a LidarLite instance connected to specified I2C pins
     *
     * @param sda I2C-bus SDA pin
     * @param scl I2C-bus SCL pin
     */
    LidarLite(PinName sda, PinName scl);

    /** Queries distance (range) and velocity registers of the sensor
     *
     */
    void refreshRangeVelocity();            
    
    /** Queries velocity register of the sensor
     *
     */
    void refreshVelocity();                 
    
    /** Queries distance (range) register of the sensor
     *
     */
    void refreshRange();                    

    /** Returns distance in cm (range) measured by sensor as read by earlier refresh function
     *
     */
    int16_t getRange_cm();
    
    /** Returns velocity in cm/s measured by sensor as read by earlier refresh function
     *
     */                  
    int16_t getVelocity_cms();              

private:
    I2C* i2c_;
    int16_t distance_LL;
    int16_t velocity_LL;
};

#endif /* LidarLite_H */