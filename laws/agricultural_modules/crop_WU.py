# -------------------------------------------------------------------------
# Name:        Crop water-uptake calculations
# Purpose:
#
# Author:      Zhonghao Fu
#
# Created:     23/09/2024
# Copyright:   (c) Zhonghao Fu 2024
# -------------------------------------------------------------------------

from laws.management_modules.globals import dateVar
import numpy as np
from laws.management_modules.data_handling import *

class crop_WU(object):
    """Calculate crop root-water uptake and daily water stress.

    **Global variables**

    ==========================================  ========================================================================  =================
    Variable [self.var]                         Description                                                               Unit
    ==========================================  ========================================================================  =================
    Bn_{Irr,nonIrr}_daily                       Daily biomass state for irrigated and rainfed crops                       t ha-1
    Crops_names                                 Names of crops represented by the crop-index dimension                   --
    EPP_{Irr,nonIrr}_daily                      Daily potential transpiration for irrigated and rainfed crops             m
    ET_crop_{Irr,nonIrr}_m                      Evapotranspiration per unit crop area for irrigated and rainfed crops      m
    GP_{Irr,nonIrr}                             Maximum growing-period duration for irrigated and rainfed crops            day
    GernalCropWaterDeficit_1                    First fitted parameter of the general-crop water-stress response          --
    GernalCropWaterDeficit_2                    Second fitted parameter of the general-crop water-stress response         --
    GernalCropWaternonDeficit                   Relative available-water threshold below which general crops are stressed --
    GrowPeriod_{Irr,nonIrr}                     Elapsed growing-period duration for irrigated and rainfed crops            day
    HD_{Irr,nonIrr}                             Harvest day of year for irrigated and rainfed crops                       day of year
    HUF_{Irr,nonIrr}_daily                      Heat-unit factor for leaf-area development for irrigated and rainfed crops --
    HUI_{Irr,nonIrr}_daily                      Accumulated heat units divided by potential heat units for both systems   --
    JJT_{Irr,nonIrr}                            Crop status code for irrigated and rainfed crops                          int
    N_stress_{Irr,nonIrr}                       Nutrition-stress multiplier for irrigated and rainfed crops                --
    PD_{Irr,nonIrr}                             Planting day of year for irrigated and rainfed crops                      day of year
    PHU_{Irr,nonIrr}                            Potential heat units to maturity for irrigated and rainfed crops          degree C day
    RD_{Irr,nonIrr}_daily                       Daily root depth for irrigated and rainfed crops                          m
    RootWU                                      Root-water-uptake distribution coefficient                              m-1
    SCRP11                                      Estimates plant water stress as a function of plant available Water stored
    SCRP21                                      Governs plant water stress as a function of soil water tension
    SPLIT_RZ                                    Thickness of each subdivided second-soil-layer segment                   m
    SPLIT_w2                                    Water storage in subdivided second-soil-layer segments                   m
    SPLIT_wfc2                                  Field-capacity water storage in subdivided second-soil-layer segments    m
    SPLIT_wres2                                 Residual water storage in subdivided second-soil-layer segments          m
    SPLIT_wwp2                                  Wilting-point water storage in subdivided second-soil-layer segments     m
    WS_{Irr,nonIrr}_daily                       Daily water-stress multiplier for irrigated and rainfed crops             --
    WSfactor                                    Heat-unit-index threshold for accumulating harvest-index transpiration   --
    WSweight                                    Weight assigned to transpiration-ratio water stress                      --
    actPYield_{Irr,nonIrr}                      Potential-yield state for irrigated and rainfed crops                     t ha-1
    actTransForHI_{Irr,nonIrr}_daily            Actual transpiration accumulated for harvest response in both systems    m
    actYield_{Irr,nonIrr}                       Harvested crop yield for irrigated and rainfed crops                      t ha-1
    act_Transpt_{Irr,nonIrr}_daily              Daily actual transpiration for irrigated and rainfed crops                m
    act_Transpt_soilLayer_{Irr,nonIrr}_daily    Daily actual transpiration by CWatM soil layer for both systems           m
    dHUF_{Irr,nonIrr}_daily                     Daily heat-unit-factor increment for irrigated and rainfed crops          --
    fracCrops_{Irr,nonIrr}                      Grid-cell fraction occupied by each irrigated and rainfed crop            --
    potTransForHI_{Irr,nonIrr}_daily            Potential transpiration accumulated for harvest response in both systems  m
    rootDepth                                   Cumulative lower boundary of each CWatM soil layer                       m
    totAvlWater                                 Total plant-available water in the soil profile                          m
    w1                                          Water storage in CWatM soil layer 1                                      m
    w2                                          Water storage in CWatM soil layer 2                                      m
    w3                                          Water storage in CWatM soil layer 3                                      m
    wfc1                                        Field-capacity water storage in CWatM soil layer 1                       m
    wfc3                                        Field-capacity water storage in CWatM soil layer 3                       m
    wres3                                       Residual water storage in CWatM soil layer 3                             m
    wwp1                                        Wilting-point water storage in CWatM soil layer 1                        m
    wwp2                                        Wilting-point water storage in CWatM soil layer 2                        m
    wwp3                                        Wilting-point water storage in CWatM soil layer 3                        m
    ==========================================  ========================================================================  =================

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.var

    def initial(self):

        SCRP21_1 = 100.01
        SCRP21_2 = 1000.90
        X1,Y1 = self.scrp_to_xy(SCRP21_1)
        X2,Y2 = self.scrp_to_xy(SCRP21_2)
        self.SCRP21_1,self.SCRP21_2 = self.calculate_s_curve_params(X1,Y1,X2,Y2)

        SCRP11_1 = 20.10
        SCRP11_2 = 50.95
        X1,Y1 = self.scrp_to_xy(SCRP11_1)
        X2,Y2 = self.scrp_to_xy(SCRP11_2)
        self.SCRP11_1,self.SCRP11_2 = self.calculate_s_curve_params(X1,Y1,X2,Y2)

        if 'GernalCropWaternonDeficit' in binding:
            self.var.GernalCropWaternonDeficit = loadmap('GernalCropWaternonDeficit')

            GernalCropWaterDeficit_1 = loadmap('GernalCropWaterDeficit_1')
            GernalCropWaterDeficit_2 = loadmap('GernalCropWaterDeficit_2')
            X1,Y1 = self.scrp_to_xy(GernalCropWaterDeficit_1)
            X2,Y2 = self.scrp_to_xy(GernalCropWaterDeficit_2)
            x1 = X1 / 100
            x2 = X2 / 100
            self.var.GernalCropWaterDeficit_1,self.var.GernalCropWaterDeficit_2 = self.calculate_s_curve_params(x1,Y1,x2,Y2)

    def calculateWU(self,RD,EPP,pm54,No):
        CPWU = 1
        SU = globals.inZero.copy()
        UX = globals.inZero.copy()
        U = np.tile(globals.inZero, (5, 1))
        SEV = np.tile(globals.inZero, (3, 1))
        # After splitting the soil profile, represent the root zone using five layers.
        SD0 = 0
        for ISL in range(5):
            # The first layer corresponds to the CWatM first soil layer.
            if ISL == 0:
                SD = self.var.rootDepth[0][No].copy()
                wwp = self.var.wwp1[No].copy()
                ST = self.var.w1[No].copy()
                wfc = self.var.wfc1[No].copy()
                wres = self.var.SPLIT_wres2[ISL-1][No].copy()
            # Layers two through four are subdivisions of the original second layer.
            elif ISL < 4:
                SD = self.var.SPLIT_RZ[ISL-1][No].copy() + SD0
                wwp = self.var.SPLIT_wwp2[ISL-1][No].copy()
                ST = self.var.SPLIT_w2[ISL-1][No].copy()
                wfc = self.var.SPLIT_wfc2[ISL-1][No].copy()
                wres = self.var.SPLIT_wres2[ISL-1][No].copy()
            # The fifth layer corresponds to the CWatM third soil layer.
            elif ISL == 4:
                SD = self.var.rootDepth[2][No].copy()
                wwp = self.var.wwp3[No].copy()
                ST = self.var.w3[No].copy()
                wfc = self.var.wfc3[No].copy()
                wres = self.var.wres3[No].copy()
            # For the five-layer soil profile, RZ is the root-zone depth of this layer.
            RZ = np.minimum(SD, 2)

            # pm54 = 5.
            UB1 = pm54*RZ

            GX = np.where(RD > SD, SD, RD)

            SUM = np.where(RD>SD0,EPP*(1.-np.exp(-UB1*GX/RD))/(1-np.exp(-UB1)),0)

            BLM = np.where(SUM>0,np.where(SD<=0.5,wres,wwp),0)
            ##IF CALCULATES ROOT GROWTH STRESSES CAUSED BY TEMPERATURE, ALUMINUM TOXICITY,
            # AND SOIL STRENGTH AND DETERMINES THE ACTIVE CONSTRAINT ON ROOT GROWTH (THE MINIMUM STRESS FACTOR).
            pm2 = 2  # Root-growth stress option.
            if pm2 == 2:
                RGS = 1
                CPWU = CPWU * RGS

            # TOS=36.*ECND(ISL)  ##ECND:ELECTRICAL COND (mmHO/CM)   set to 0.3 here
            TOS = 36 * 0.3

            WTN = np.where(SUM>0,np.maximum(5., 10.**(3.1761 - 1.6576 * ((np.log10(ST) - np.log10(wwp)) / (np.log10(wfc) - np.log10(wwp))))),0)

            XX = TOS + WTN

            F = np.where(SUM>0,1. - XX / (XX + np.exp(self.SCRP21_1 - self.SCRP21_2 * XX)),0)  ##

            U[ISL] = np.where(SUM>0,np.minimum(SUM-CPWU*SU-(1.-CPWU)*UX,ST-BLM)*RGS*F,0)
            U[ISL] = np.where(XX<5000,U[ISL],0)
            U[ISL] = np.maximum(U[ISL],0)

            SU += U[ISL]
            UX = SUM

            SD0 = SD

        SEV[0] = U[0]
        SEV[1] = U[1] + U[2] + U[3]
        SEV[2] = U[4]
        return SU,SEV


    def dynamic(self):
            # Only go through this once:
            # I. new start
        if dateVar['newStart']:

            for z in ['HUI_Irr_daily','HUI_nonIrr_daily','HUF_Irr_daily','HUF_nonIrr_daily',
                        'dHUF_Irr_daily','dHUF_nonIrr_daily','Bn_Irr_daily','Bn_nonIrr_daily','actPYield_Irr','actPYield_nonIrr','RD_Irr_daily','RD_nonIrr_daily','GrowPeriod_Irr','GrowPeriod_nonIrr',\
                        'fracCrops_Irr','fracCrops_nonIrr','JJT_Irr','JJT_nonIrr',
                        'PD_Irr','PD_nonIrr','GP_Irr','GP_nonIrr','HD_Irr','HD_nonIrr','PHU_Irr','PHU_nonIrr',
                        'N_stress_Irr','N_stress_nonIrr','ET_crop_Irr_m','ET_crop_nonIrr_m','actYield_Irr','actYield_nonIrr']:
                vars(self.var)[z] = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            crop_vari = ['ET_crop_Irr_m','ET_crop_nonIrr_m','actYield_Irr','actYield_nonIrr']
            for crop_z in crop_vari:
                for i in range(len(self.var.Crops_names)):
                    vars(self.var)[crop_z+str(i)] = globals.inZero.copy()

            self.var.fracCrops_Irr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            self.var.fracCrops_nonIrr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))

            for i in range(len(self.var.Crops_names)):
                cropfile = cbinding(self.var.Crops_names[i] + '_Irr')
                self.var.PD_Irr[i] = readnetcdfWithoutTime(cropfile, value='planting_date')
                self.var.PD_nonIrr[i] = readnetcdfWithoutTime(cropfile, value='planting_date')

                self.var.GP_Irr[i] = readnetcdfWithoutTime(cropfile,value='growing_period')
                self.var.GP_nonIrr[i] = readnetcdfWithoutTime(cropfile,value='growing_period')

                self.var.HD_Irr[i] = readnetcdfWithoutTime(cropfile, value='harvest_date')
                self.var.HD_nonIrr[i] = readnetcdfWithoutTime(cropfile, value='harvest_date')

                self.var.PHU_Irr[i] = readnetcdfWithoutTime(cropfile, value='phu')
                self.var.PHU_nonIrr[i] = readnetcdfWithoutTime(cropfile, value='phu')

                self.var.N_stress_Irr[i] = readnetcdfWithoutTime(cropfile, value='N_stress')
                self.var.N_stress_nonIrr[i] = readnetcdfWithoutTime(cropfile, value='N_stress')
                if str('site_sim') in binding:
                    if returnBool("site_sim"):
                        if str(self.var.Crops_names[i] + '_PD') in binding:
                            self.var.PD_Irr[i] = int(binding[self.var.Crops_names[i] + '_PD'])
                            self.var.PD_nonIrr[i] = int(binding[self.var.Crops_names[i] + '_PD'])
                            self.var.GP_Irr[i] = int(binding[self.var.Crops_names[i] + '_GP'])
                            self.var.GP_nonIrr[i] = int(binding[self.var.Crops_names[i] + '_GP'])
                            self.var.HD_Irr[i] = int(binding[self.var.Crops_names[i] + '_HD'])
                            self.var.HD_nonIrr[i] = int(binding[self.var.Crops_names[i] + '_HD'])
                            self.var.PHU_Irr[i] = int(binding[self.var.Crops_names[i] + '_PHU'])
                            self.var.PHU_nonIrr[i] = int(binding[self.var.Crops_names[i] + '_PHU'])

        for c in range(len(self.var.Crops_names)):
            pm3 = self.var.WSfactor[c]
            pm54 = self.var.RootWU[c]

            ##for nonIrr
            calc_result_nonirr = self.calculateWU(self.var.RD_nonIrr_daily[c],self.var.EPP_nonIrr_daily[c],pm54,No=1)
            self.var.act_Transpt_nonIrr_daily[c] = np.where(self.var.EPP_nonIrr_daily[c]>0, calc_result_nonirr[0], 0)
            self.var.act_Transpt_soilLayer_nonIrr_daily[c] = np.where(self.var.EPP_nonIrr_daily[c]>0, calc_result_nonirr[1], 0)

            self.var.actTransForHI_nonIrr_daily[c] += np.where((self.var.HUI_nonIrr_daily[c]>pm3)&(self.var.HUI_nonIrr_daily[c]<=1),self.var.act_Transpt_nonIrr_daily[c],0)

            self.var.potTransForHI_nonIrr_daily[c] += np.where((self.var.HUI_nonIrr_daily[c]>pm3)&(self.var.HUI_nonIrr_daily[c]<=1),self.var.EPP_nonIrr_daily[c],0)

            ##for nonIrr
            No = 1
            availWaterPlant1 = self.var.w1[No] - self.var.wwp1[No] #  Root zone water deficiency is allowed
            availWaterPlant2 = self.var.w2[No] - self.var.wwp2[No] # *
            availWaterPlant3 = self.var.w3[No] - self.var.wwp3[No]  # *
            readAvlWater = availWaterPlant1 + availWaterPlant2 + availWaterPlant3  # + availWaterPlant3

            WStemp = np.where(readAvlWater>0,100*readAvlWater/self.var.totAvlWater,0)
            WStemp1 = WStemp/(WStemp+np.exp(self.SCRP11_1-self.SCRP11_2*WStemp))
            pm35 = self.var.WSweight[c]
            # pm35 = 1
            self.var.WS_nonIrr_daily[c] = np.where(self.var.EPP_nonIrr_daily[c]>0,\
                                                            np.where(readAvlWater>0,(1-pm35)*WStemp1+pm35*self.var.act_Transpt_nonIrr_daily[c]/(self.var.EPP_nonIrr_daily[c]),0),1)
            ##for Irr
            if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                ##for Irr-nonPaddy
                calc_result_irr = self.calculateWU(self.var.RD_Irr_daily[c],self.var.EPP_Irr_daily[c],pm54,No=3)
                self.var.act_Transpt_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0, calc_result_irr[0], 0)
                self.var.act_Transpt_soilLayer_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0, calc_result_irr[1], 0)

                self.var.actTransForHI_Irr_daily[c] += np.where((self.var.HUI_Irr_daily[c]>pm3)&(self.var.HUI_Irr_daily[c]<=1),self.var.act_Transpt_Irr_daily[c],0)

                self.var.potTransForHI_Irr_daily[c] += np.where((self.var.HUI_Irr_daily[c]>pm3)&(self.var.HUI_Irr_daily[c]<=1),self.var.EPP_Irr_daily[c],0)

                No = 3
                availWaterPlant1 = self.var.w1[No] - self.var.wwp1[No] #  Root zone water deficiency is allowed
                availWaterPlant2 = self.var.w2[No] - self.var.wwp2[No] # *
                availWaterPlant3 = self.var.w3[No] - self.var.wwp3[No]  # *
                readAvlWater = availWaterPlant1 + availWaterPlant2 + availWaterPlant3  # + availWaterPlant3
                WStemp = np.where(readAvlWater>0,100*readAvlWater/self.var.totAvlWater,0)
                WStemp1 = WStemp/(WStemp+np.exp(self.SCRP11_1-self.SCRP11_2*WStemp))
                self.var.WS_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0,\
                                                                np.where(readAvlWater>0,(1-pm35)*WStemp1+pm35*self.var.act_Transpt_Irr_daily[c]/(self.var.EPP_Irr_daily[c]),0),1)
            else:
                ##for Irrpaddy
                calc_result_irr = self.calculateWU(self.var.RD_Irr_daily[c],self.var.EPP_Irr_daily[c],pm54,No=2)
                self.var.act_Transpt_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0, calc_result_irr[0], 0)
                self.var.act_Transpt_soilLayer_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0, calc_result_irr[1], 0)

                self.var.actTransForHI_Irr_daily[c] += np.where((self.var.HUI_Irr_daily[c]>pm3)&(self.var.HUI_Irr_daily[c]<=1),self.var.act_Transpt_Irr_daily[c],0)

                ##for Irr-Paddy
                No = 2
                availWaterPlant1 = self.var.w1[No] - self.var.wwp1[No] #  Root zone water deficiency is allowed
                availWaterPlant2 = self.var.w2[No] - self.var.wwp2[No] # *
                availWaterPlant3 = self.var.w3[No] - self.var.wwp3[No]  # *
                readAvlWater = availWaterPlant1 + availWaterPlant2 + availWaterPlant3  # + availWaterPlant3
                WStemp = np.where(readAvlWater>0,100*readAvlWater/self.var.totAvlWater,0)
                WStemp1 = WStemp/(WStemp+np.exp(self.SCRP11_1-self.SCRP11_2*WStemp))
                self.var.WS_Irr_daily[c] = np.where(self.var.EPP_Irr_daily[c]>0,\
                                                                np.where(readAvlWater>0,(1-pm35)*WStemp1+pm35*self.var.act_Transpt_Irr_daily[c]/(self.var.EPP_Irr_daily[c]),0),1)

    def scrp_to_xy(self,scrp_point):
        """
        Decode an SCRP-formatted value (for example, 15.01) into
        its X and Y components (X=15, Y=0.01).
        """
        X = int(scrp_point)  # Use the integer part as X.
        Y = round(scrp_point - int(scrp_point), 2)  # Use the fractional part as Y.
        return X, Y

    def calculate_s_curve_params(self,X1,Y1,X2,Y2):
        """
        Calculate B1 and B2 from two decoded SCRP control points.

        :param X1: X-coordinate of the first control point.
        :param Y1: Y-coordinate of the first control point.
        :param X2: X-coordinate of the second control point.
        :param Y2: Y-coordinate of the second control point.
        :return: B1 and B2 parameters of the S-curve.
        """


        # The input control points have already been decoded.
        # X1, Y1 = self.scrp_to_xy(scrp1)
        # X2, Y2 = self.scrp_to_xy(scrp2)

        xx = np.log(X1 /Y1-X1)
        B2 = (xx - np.log(X2 / Y2 - X2)) / (X2 - X1)
        B1 = xx + X1 * B2

        return np.round(B1, 6), np.round(B2, 6)
