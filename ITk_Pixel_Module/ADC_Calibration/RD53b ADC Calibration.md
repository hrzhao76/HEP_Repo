# RD53b ADC Calibration

---
Code development repo: 
[itkpix-module-qc](https://gitlab.cern.ch/berkeleylab/itk-pixel-module-software/itkpix-module-qc/-/tree/master/)

TODO: 
1. build a frame, write adc calibration codes using the json file, mimicing what is done for IV curve 
2. discuss with Yarr developer and figure out the interface with Yarr, e.g. `write-register`

# RD53a ADC Calibration reference

1. Set the chip configuration file:
`configs/rd53a_test.json`
    
    ```json
    "InjVcalDiff": 3000
    "InjVcalHigh": 3500
    "InjVcalMed": 500
    "MonitorEnable": 1
    "MonitorVmonMux": 1 # VMux = 1, CAL_MED is selected
    
    # There is some previous uncalibrated data
    "ADCcalPar": [5.894350051879883, 0.1920430064201355]
    
    ```
    
    ![Screenshot_20220801_114809.png](RD53b%20ADC%20Calibration%207c198708b98148fa8458fa6b6dd613a9/Screenshot_20220801_114809.png)
    
2. Run the chip configuration, and measure the `Vcal_med` through meter on the board pins
3. Change the chip configuration file `configs/rd53a_test.json`
    
    ```json
    "InjVcalDiff": 3000
    "InjVcalHigh": 3500
    "InjVcalMed": 500
    "MonitorEnable": 1
    "MonitorVmonMux": 2 # VMux = 2, CAL_HI is selected
    ```
    
4. Run the chip configuration, and measure the `Vcal_high` through meter on the board pins
5. Set the scan file `configs/scans/rd53a/reg_readmux.json`
    
    ```json
    "InjVcalHigh": 3500
    "InjVcalMed": 500
    
    # Notice there is a loop above, which run through all the mux selections
    "loops": [
            {
                "config": {
                    "VoltMux": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                    "CurMux": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                    "Registers": ["Null"],
                    "EnblRingOsc": 0,
                    "RingOscRep": 10,
                    "RingOscDur": 9
    
                },
                "loopAction": "Rd53aReadRegLoop"
            },
    
    ```
    
6. Run the scan and get the ADC output
    
    ```json
    [15:34:50:135][ info ][Rd53aReadRegLoop]: [0][14195] MON MUX_V: 1, Value: 454 => 0.0982428 V
    [15:34:50:137][ info ][Rd53aReadRegLoop]: [0][14195] MON MUX_V: 2, Value: 3574 => 0.701027 V
    ```
    
7. Fit a line with `(MON MUX_V1_value , Vcal_med) and (MON MUX_V2_value , Vcal_high)`
Get the offset and the slope 
8. Enter the offset and slope (both in mV) into the chip configuration file: 
`"ADCcalPar": [<offset>, <slope>]`

# RD53b

## Naming Difference

### Global Config

`configs/rd53b_test.json`

```json

{
    "RD53B": {
        "GlobalConfig": {
            //.......
            "MonitorEnable": 0,
            "MonitorI": 63, // IMux 63 not used 
            "MonitorV": 33, // VINA/4
            "MonitoringDataAdc": 0,
            //.......
```

![Screenshot_20220801_120742.png](RD53b%20ADC%20Calibration%207c198708b98148fa8458fa6b6dd613a9/Screenshot_20220801_120742.png)

### ADCcalPar

`configs/rd53b_test.json`

```json
"Parameter": {
            "ADCcalPar": [5.894350051879883, 0.1920430064201355, 4990.0],
            "ChipId": 15,
            "EnforceNameIdCheck": true,
            "InjCap": 7.5,
            "MosCalPar": 1.2640000581741334,
            "Name": "0x16a4c",
            "NtcCalPar": [0.0007488999981433153, 0.0002769000129774213, 7.059500006789676e-8],
            "VcalPar": [0.46000000834465029, 0.20069999992847444]
        },
```

```json
std::array<float, 3> m_adcCalPar; //mV, [0] + [1]*x, R_IMUX = [2]
```

## Global Config Initilization

```cpp

//99
    MonitorEnable.init      ( 99, &m_cfg[ 99], 12,  1, 0); regMap["MonitorEnable"] = &Rd53bGlobalCfg::MonitorEnable;
    MonitorI.init           ( 99, &m_cfg[ 99], 6,  6, 63); regMap["MonitorI"] = &Rd53bGlobalCfg::MonitorI;
    MonitorV.init           ( 99, &m_cfg[ 99], 0,  6, 63); regMap["MonitorV"] = &Rd53bGlobalCfg::MonitorV;
```

# Reference function codes

```cpp
// src/libRd53b/Rd53b.cpp
void Rd53b::confAdc(uint16_t MONMUX, bool doCur) {
    //This only works for voltage MUX values.
    uint16_t OriginalGlobalRT = this->GlobalPulseConf.read();
    uint16_t OriginalMonitorEnable = this->MonitorEnable.read(); //Enabling monitoring
    uint16_t OriginalMonitorV = this->MonitorV.read();
    uint16_t OriginalMonitorI = this->MonitorI.read();

    if (doCur)
    {
        this->writeRegister(&Rd53b::MonitorV, 1);      // Forward via VMUX
        this->writeRegister(&Rd53b::MonitorI, MONMUX); // Select what to monitor
    }
    else
    {
        this->writeRegister(&Rd53b::MonitorV, MONMUX); // Select what to monitor
    }

    this->writeRegister(&Rd53b::MonitorEnable, 1); // Enabling monitoring
    while(!core->isCmdEmpty()){;}

    std::this_thread::sleep_for(std::chrono::microseconds(100));

    this->writeRegister(&Rd53b::GlobalPulseConf, 0x40); // Reset ADC
    this->writeRegister(&Rd53b::GlobalPulseWidth, 4);   // Duration = 4 inherited from RD53A
    while(!core->isCmdEmpty()){;}
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    this->sendGlobalPulse(m_chipId);
    std::this_thread::sleep_for(std::chrono::microseconds(1000000)); // Need to wait long enough for ADC to reset

    this->writeRegister(&Rd53b::GlobalPulseConf, 0x1000); //Trigger ADC Conversion
    while (!core->isCmdEmpty()){;}
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    this->sendGlobalPulse(m_chipId);
    std::this_thread::sleep_for(std::chrono::microseconds(1000)); //This is neccessary to clean. This might be controller dependent.

    // Reset register values
    this->writeRegister(&Rd53b::GlobalPulseConf, OriginalGlobalRT);
    this->writeRegister(&Rd53b::MonitorEnable, OriginalMonitorEnable);
    this->writeRegister(&Rd53b::MonitorV, OriginalMonitorV);
    this->writeRegister(&Rd53b::MonitorI, OriginalMonitorI);
    while (!core->isCmdEmpty()){;}
    std::this_thread::sleep_for(std::chrono::microseconds(100));
}
```

```cpp
// src/libRd53b/Rd53bReadRegLoop.cpp
//Configures the ADC, reads the register returns the first recieved register.
uint16_t Rd53bReadRegLoop::ReadADC(unsigned short Reg, bool doCur, Rd53b *tmpFE)
{

    if (tmpFE == NULL)
        tmpFE = keeper->globalFe<Rd53b>();

    g_tx->setCmdEnable(dynamic_cast<FrontEndCfg *>(tmpFE)->getTxChannel());
    tmpFE->confAdc(Reg, doCur);
    g_tx->setCmdEnable(keeper->getTxMask());

    uint16_t RegVal = ReadRegister(&Rd53b::MonitoringDataAdc, dynamic_cast<Rd53b *>(tmpFE));

    return RegVal;
}
```

```cpp
// src/libRd53b/Rd53bReadRegLoop.cpp
// Reading Voltage  ADC
            for (auto Reg : m_VoltMux)
            {
                uint16_t ADCVal = ReadADC(Reg, false, feRd53b);
                logger->info("[{}][{}] MON MUX_V: {}, Value: {} => {} V", id, feName, Reg, ADCVal, dynamic_cast<Rd53b *>(fe)->adcToV(ADCVal));
            }

// Reading Current ADC
            for (auto Reg : m_CurMux)
            {
                uint16_t ADCVal = ReadADC(Reg, true, feRd53b);
                logger->info("[{}][{}] MON MUX_C: {} Value: {} => {} uA", id, feName, Reg, ADCVal, dynamic_cast<Rd53b *>(fe)->adcToI(ADCVal)/1e-6);
            }

```

# Proposal Procedure

1. write a standalone c++ file for SCC: 

```cpp
fe -> WriteNamedRegister("MonitorEnable", 1); // Enable monitoring
fe -> WriteNamedRegister("MonitorV", 8); // VCAL_MED
// InjVcalMed, InjVcalHigh
RegVal = ReadRegister(&Rd53b::MonitoringDataAdc, dynamic_cast<Rd53b *>(fe)) // ADC counts 
meter -> measCurrent();

fe -> WriteNamedRegister("MonitorV", 7); // VCAL_HI
// InjVcalMed, InjVcalHigh
RegVal = ReadRegister(&Rd53b::MonitoringDataAdc, dynamic_cast<Rd53b *>(fe)) // ADC counts 
meter -> measCurrent(); // 

// calculate offset slope... 
```

1. embedded with json file… use python to call?  

# Question

1. how to deal with `InjVcalHigh`  without scanConsole? 

```bash
./bin/powersupply -n PS3 -c 1 -e ../src/configs/input-hw.json power-on -d

bin/scanConsole -r configs/controller/specCfg-rd53b-16x1.json -c configs/connectivity/example_rd53b_setup.json

./bin/meter --equip ../meter.json -n Fluke45 meas-voltage -d
```