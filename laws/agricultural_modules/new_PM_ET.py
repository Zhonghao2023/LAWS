# -------------------------------------------------------------------------
# Name:        Crop-specific evapotranspiration
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

class new_PM_ET(object):
    """Calculate crop-specific potential transpiration and soil evaporation.

    **Global variables**

    ==========================================  ========================================================================  =================
    Variable [self.var]                         Description                                                               Unit
    ==========================================  ========================================================================  =================
    CHMX_{Irr,nonIrr}                           Running maximum height for irrigated and rainfed crops                    m
    Crops_names                                 Names of crops represented by the crop-index dimension                   --
    EAct                                        Actual vapour pressure                                                    kPa
    EO_with_{Irr,nonIrr}crop                    Potential evapotranspiration before partitioning for both crop systems     m
    EO_without_{Irr,nonIrr}crop                 Potential evapotranspiration of uncropped irrigated and rainfed land       m
    EO_without_Irrpaddy                         Potential evapotranspiration of irrigated paddy land without a crop         m
    EPP_{Irr,nonIrr}_daily                      Daily potential transpiration for irrigated and rainfed crops             m
    ES_with_{Irr,nonIrr}crop                    Daily actual soil evaporation beneath irrigated and rainfed crops         m
    ES_with_{Irr,nonIrr}crop_layer              Daily actual soil evaporation by layer for both crop systems             m
    ES_without_{Irr,nonIrr}crop                 Actual soil evaporation from uncropped irrigated and rainfed land        m
    ES_without_{Irr,nonIrr}crop_layer           Soil-layer contribution to evaporation from uncropped land              m
    ES_without_Irrpaddy                         Actual soil evaporation from irrigated paddy land without a crop           m
    ES_without_Irrpaddy_layer                   Soil-layer contribution to irrigated paddy evaporation without a crop      m
    ES_without_crop_layer                       Soil-layer evaporation from irrigated fallow land                         m
    ET_correct                                  Penman-Monteith evapotranspiration adjustment factor                       --
    EV_LAI                                      LAI-dependent soil-evaporation attenuation coefficient                    --
    EV_correct                                  Soil-water control coefficient for actual evaporation                     --
    EVweight                                    Weight controlling evaporation extraction across soil layers              --
    GSM                                         Maximum stomatal conductance                                              m s-1
    HCM                                         Maximum crop height                                                       m
    JJT_{Irr,nonIrr}                            Crop status code for irrigated and rainfed crops                          int
    LAI_{Irr,nonIrr}_daily                      Leaf area per unit ground area for irrigated and rainfed crops            m2 m-2
    Psurf                                       Surface air pressure used by the crop evapotranspiration routine          kPa
    REG_{Irr,nonIrr}_daily                      Biomass-growth stress multiplier for irrigated and rainfed crops          --
    RootWU                                      Root-water-uptake distribution coefficient                               m-1
    Rsdl                                        Surface-downwelling longwave radiation                                   MJ m-2 day-1
    Rsds                                        Surface-downwelling shortwave radiation                                  MJ m-2 day-1
    Rto1                                        Fraction of CWatM soil-layer-2 depth assigned to its first subdivision    --
    Rto2                                        Fraction assigned to each remaining soil-layer-2 subdivision              --
    SCRP2                                       Encoded control points for the soil-depth evaporation response            --
    SnowCover                                   Snow-water-equivalent storage                                             m
    SPLIT_RZ                                    Thickness of each subdivided second-soil-layer segment                    m
    SPLIT_w2                                    Water storage in subdivided second-soil-layer segments                    m
    SPLIT_wfc2                                  Field-capacity water storage in subdivided second-soil-layer segments     m
    SPLIT_wres2                                 Residual water storage in subdivided second-soil-layer segments           m
    SPLIT_wwp2                                  Wilting-point water storage in subdivided second-soil-layer segments      m
    STLBm_{Irr,nonIrr}_daily                    Above-ground biomass for irrigated and rainfed crops                     t ha-1
    TMax                                        Daily maximum air temperature                                            degree C
    TMin                                        Daily minimum air temperature                                            degree C
    TS_daily                                    Daily temperature-stress multiplier                                      --
    Tavg                                        Daily mean air temperature                                               degree C
    VPC1                                        Encoded VPD conductance response point                                    --
    VPT                                         Vapour-pressure-deficit threshold for stomatal response                   kPa
    WS_{Irr,nonIrr}_daily                       Daily water-stress multiplier for irrigated and rainfed crops             --
    WSfactor                                    Heat-unit-index threshold for accumulating harvest-index transpiration    --
    WSweight                                    Weight assigned to transpiration-ratio water stress                       --
    Wind                                        Wind speed supplied at 2 m                                                m s-1
    actTransForHI_{Irr,nonIrr}_daily            Actual transpiration accumulated for harvest response in both systems     m
    actYield_{Irr,nonIrr}                       Harvested crop yield for irrigated and rainfed crops                      t ha-1
    act_Transpt_{Irr,nonIrr}_daily              Daily actual transpiration for irrigated and rainfed crops                m
    act_Transpt_soilLayer_{Irr,nonIrr}_daily    Daily actual transpiration by CWatM soil layer for both systems           m
    albedoLand                                  Land-surface albedo                                                       --
    canopy                                      Crop-canopy resistance adjustment parameter                               --
    co2                                         Atmospheric carbon-dioxide concentration                                 ppm
    coverTypes                                  Names of the CWatM land-cover classes                                    --
    height                                      Generic crop-height state retained for crop-module compatibility          m
    height_{Irr,nonIrr}_daily                   Daily crop height for irrigated and rainfed crops                         m
    interceptEvap                               Evaporation from water intercepted by vegetation                          m
    potTransForHI_{Irr,nonIrr}_daily            Potential transpiration accumulated for harvest response in both systems  m
    rootDepth                                   Cumulative lower boundary of each CWatM soil layer                        m
    totalBm_{Irr,nonIrr}_daily                  Total biomass for irrigated and rainfed crops                            t ha-1
    w1                                          Water storage in CWatM soil layer 1                                       m
    w2                                          Water storage in CWatM soil layer 2                                       m
    wfc1                                        Field-capacity water storage in CWatM soil layer 1                        m
    wres1                                       Residual water storage in CWatM soil layer 1                              m
    wwp1                                        Wilting-point water storage in CWatM soil layer 1                         m
    ==========================================  ========================================================================  =================

    **Functions**
    """

    def __init__(self, model):
        self.model = model
        self.var = model.var

    def initial(self):
        self.var.co2 = 330
        # Fit the S-curve that distributes potential evaporation with depth.
        SCRP2_1 = 10.50
        SCRP2_2 = 100.95

        X1,Y1 = self.scrp_to_xy(SCRP2_1)
        X2,Y2 = self.scrp_to_xy(SCRP2_2)
        self.SCRP2_1,self.SCRP2_2 = self.calculate_s_curve_params(X1,Y1,X2,Y2)
        for z in ['canopy','WSfactor','EV_correct','WSweight','EV_LAI','EVweight','RootWU','ET_correct']:
            vars(self.var)[z] = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
        for i in range(len(self.var.Crops_names)):
            if str(self.var.Crops_names[i]+'_param') in binding:
                cropfile = cbinding(self.var.Crops_names[i]+'_param')
                self.var.canopy[i] = readnetcdfWithoutTime(cropfile, value='canopy')
                self.var.WSfactor[i] = readnetcdfWithoutTime(cropfile, value='WSfactor')
                self.var.EV_correct[i] = readnetcdfWithoutTime(cropfile, value='EV_correct')
                self.var.WSweight[i] = readnetcdfWithoutTime(cropfile, value='WSweight')
                self.var.EV_LAI[i] = readnetcdfWithoutTime(cropfile, value='EV_LAI')
                self.var.EVweight[i] = readnetcdfWithoutTime(cropfile, value='EVweight')
                self.var.RootWU[i] = readnetcdfWithoutTime(cropfile, value='RootWU')
                self.var.ET_correct[i] = readnetcdfWithoutTime(cropfile, value='ET_correct')
            else:
                self.var.canopy[i] = float(binding['canopy'])
                self.var.WSfactor[i] = float(binding['WSfactor'])
                self.var.EV_correct[i] = float(binding['EV_correct'])
                self.var.WSweight[i] = float(binding['WSweight'])
                self.var.EV_LAI[i] = float(binding['EV_LAI'])
                self.var.EVweight[i] = float(binding['EVweight'])
                self.var.RootWU[i] = float(binding['RootWU'])
                self.var.ET_correct[i] = float(binding['ET_correct'])

    def calculateZZZDRV(self,U10,CHMX,HCM):
        """Adjust wind speed to a crop-dependent reference height.

        :param U10: Wind speed at 10 m, in m s-1.
        :param CHMX: Running maximum crop height, in m.
        :param HCM: Maximum crop height parameter, in m.
        :return: ``ZZ`` [m], the wind reference height, and ``UZZ``
            [m s-1], the wind speed adjusted to that height.
        """
        # Keep the 10 m reference for ordinary crops; measure above tall canopies.
        ZZ = np.where(CHMX > 8,HCM + 2,10)
        UZZ = np.where(CHMX > 8,U10 * np.log(ZZ/.0005)/9.9035,U10)
        return ZZ,UZZ

    def calculate_act_ES(self,XX,pm12,pm61,No):
        ES = 0
        TOT = 0
        SD0 = 0
        SEV = np.tile(globals.inZero, (2, 1))
        # Although splitting can produce up to four soil layers, evaporation
        # is calculated only for soil shallower than 0.2 m.
        for ISL in range(2):
            if ISL == 0:
                SD = self.var.rootDepth[0][No]
                ST = self.var.w1[No]
                wwp = self.var.wwp1[No]
                wfc = self.var.wfc1[No]
                wres = self.var.wres1[No]
            elif ISL == 1:
                SD = self.var.SPLIT_RZ[0][No] + SD0
                ST = self.var.SPLIT_w2[0][No]
                # wfc = 0.2576 - 0.002*SA + 0.0036 * CL + 0.0299*OC
                # wwp = 0.026 + 0.005*CL + 0.0158*OC
                wwp = self.var.SPLIT_wwp2[0][No]
                wfc = self.var.SPLIT_wfc2[0][No]
                wres = self.var.SPLIT_wres2[0][No]

            RTO = 1000 * SD
            # SUM is the cumulative potential evaporation accessible to this depth.
            SUM = XX * RTO / (RTO + np.exp(self.SCRP2_1 - self.SCRP2_2 * RTO))
            XZ = wfc - wwp

            F = np.where(ST<wfc,np.exp(pm12*(ST-wfc)/XZ),1)

            # Here ZZ is the incremental evaporation demand assigned to this layer.
            ZZ = SUM - pm61 * TOT - (1-pm61) * ES
            SEV[ISL] = ZZ * F


            # Update evaporation from the current soil layer.
            SEV[ISL] = np.maximum(0,np.where((ST-SEV[ISL] < wres), ST-wres, SEV[ISL]))
            # Update total soil evaporation.
            ES = ES + SEV[ISL]

            # Update cumulative potential evaporation for the next layer.
            TOT = SUM
            SD0 = SD
        return ES,SEV

    def dynamic(self):

        # I. new start
        if dateVar['newStart']:

            for z in ['EPP_Irr_daily','EPP_nonIrr_daily','LAI_Irr_daily','LAI_nonIrr_daily','height','ES_with_Irrcrop','ES_with_nonIrrcrop','ES_without_Irrcrop','ES_without_nonIrrcrop','act_Transpt_Irr_daily',\
                      'WS_Irr_daily','WS_nonIrr_daily','TS_daily','REG_Irr_daily','REG_nonIrr_daily','totalBm_Irr_daily','totalBm_nonIrr_daily','STLBm_Irr_daily','STLBm_nonIrr_daily',\
                        'EO_with_Irrcrop','EO_with_nonIrrcrop','ES_with_Irrcrop','ES_with_nonIrrcrop','CHMX_Irr','CHMX_nonIrr','height_Irr_daily','height_nonIrr_daily',
                        'JJT_Irr','JJT_nonIrr','act_Transpt_Irr_daily','act_Transpt_nonIrr_daily',\
                        'act_Transpt_soilLayer_Irr_daily','act_Transpt_soilLayer_nonIrr_daily','actTransForHI_Irr_daily','actTransForHI_nonIrr_daily','potTransForHI_Irr_daily','potTransForHI_nonIrr_daily',\
                        'actYield_Irr','actYield_nonIrr']:
                vars(self.var)[z] = np.tile(globals.inZero, (len(self.var.Crops_names), 1))

            self.var.ES_with_Irrcrop_layer = np.tile(globals.inZero, (len(self.var.Crops_names),2, 1))
            self.var.ES_with_nonIrrcrop_layer = np.tile(globals.inZero, (len(self.var.Crops_names),2, 1))
            self.var.ES_without_crop_layer = np.tile(globals.inZero, (2, 1))
            self.var.ES_without_nonIrrcrop_layer = np.tile(globals.inZero, (2, 1))
            self.CHMX_Irr =  np.tile(globals.inZero, (len(self.var.Crops_names), 1))
            self.CHMX_nonIrr =  np.tile(globals.inZero, (len(self.var.Crops_names), 1))

            self.var.act_Transpt_soilLayer_Irr_daily = np.tile(globals.inZero, (len(self.var.Crops_names), 3, 1))
            self.var.act_Transpt_soilLayer_nonIrr_daily = np.tile(globals.inZero, (len(self.var.Crops_names), 3, 1))

        # Prepare the atmospheric and radiation terms used by Penman-Monteith.
        RHO = 0.01276 * self.var.Psurf / (1.+.00367*self.var.Tavg)
        ESatmin = 0.6108* np.exp((17.27 * self.var.TMin) / (self.var.TMin + 237.3))
        ESatmax = 0.6108* np.exp((17.27 * self.var.TMax) / (self.var.TMax + 237.3))
        ESat = (ESatmin + ESatmax) / 2.0   # [KPa]
        DLT = ((4098.0 * ESat) / ((self.var.Tavg + 237.3)**2))
        RNup = 4.903E-9 * (((self.var.TMin + 273.16) ** 4) + ((self.var.TMax + 273.16) ** 4)) / 2
        RLN = RNup - self.var.Rsdl
        Rn =np.maximum(((1 - self.var.albedoLand) * self.var.Rsds - RLN), 0.0)
        X2 = Rn * DLT
        VPD = np.maximum(ESat - self.var.EAct, 0.0)
        U10 = self.var.Wind * (10/2)**0.16  # Convert 2 m wind speed to 10 m.

        GMA = 0.665E-3 * self.var.Psurf
        XL = 2.501 - 0.002361 * self.var.Tavg

        RV = 350 / U10
        pm74 = self.var.ET_correct[3]
        EO1=pm74 * 1e-3 *(X2+86.66*RHO*VPD/RV)/(XL*(DLT + GMA))  #[m d-1]

        No = 3
        EO_Irr = np.maximum(0.,EO1-self.var.interceptEvap[No])
        self.var.EO_without_Irrcrop = EO_Irr

        No = 1
        EO_nonIrr = np.maximum(0.,EO1-self.var.interceptEvap[No])
        self.var.EO_without_nonIrrcrop = EO_nonIrr

        No = 2
        EO_Irrpaddy = np.maximum(0.,EO1-self.var.interceptEvap[No])
        self.var.EO_without_Irrpaddy = EO_Irrpaddy

        i = 0
        # Synchronize the subdivided soil storages before evaporation extraction.
        for coverType in self.var.coverTypes[:4]:
            if coverType != 'forest':
                if "MinDepth_SPLIT_SOIL_LAYER" in binding:
                    self.var.SPLIT_w2[0][i] = self.var.w2[i] * self.var.Rto1[i]
                    self.var.SPLIT_w2[1][i] = self.var.w2[i] * self.var.Rto2[i]
                    self.var.SPLIT_w2[2][i] = self.var.w2[i] * self.var.Rto2[i]
            i += 1

        EAJ = np.where(self.var.SnowCover <= 0.005, np.exp(-0.04),0.5)
        #for irrnopaddy
        XX = self.var.EO_without_Irrcrop * EAJ
        act_ES_without_Irrcrop,SEV_without_Irrcrop = self.calculate_act_ES(XX,pm12=2.5,pm61=1.,No=3)
        self.var.ES_without_Irrcrop = act_ES_without_Irrcrop
        self.var.ES_without_Irrcrop_layer = SEV_without_Irrcrop

        #for nonIrrnonpaddy
        XX = self.var.EO_without_nonIrrcrop * EAJ
        act_ES_without_nonIrrcrop,SEV_without_nonIrrcrop = self.calculate_act_ES(XX,pm12=2.5,pm61=1.,No=1)
        self.var.ES_without_nonIrrcrop = act_ES_without_nonIrrcrop

        self.var.ES_without_nonIrrcrop_layer = SEV_without_nonIrrcrop

        #for irrpaddy
        XX = self.var.EO_without_Irrpaddy * EAJ
        act_ES_without_Irrpaddy,SEV_without_Irrpaddy = self.calculate_act_ES(XX,pm12=2.5,pm61=1.,No=2)
        self.var.ES_without_Irrpaddy = act_ES_without_Irrpaddy
        self.var.ES_without_Irrpaddy_layer = SEV_without_Irrpaddy

        for c in range(len(self.var.Crops_names)):
            self.CHMX_Irr[c] = np.where(self.var.height_Irr_daily[c]> self.CHMX_Irr[c],self.var.height_Irr_daily[c],self.CHMX_Irr[c])
            self.CHMX_nonIrr[c] = np.where(self.var.height_nonIrr_daily[c]> self.CHMX_nonIrr[c],self.var.height_nonIrr_daily[c],self.CHMX_nonIrr[c])

            # Calculate crop-specific aerodynamic resistance for both systems.
            ZZ,UZZ = self.calculateZZZDRV(U10,self.CHMX_Irr[c],self.var.HCM[c])
            Z0 = 0.131 * (self.CHMX_Irr[c]+0.01) ** 0.997
            ZD = 0.702 * (self.CHMX_Irr[c]+0.01) ** 0.979
            RV_Irr = 6.25*(np.log((ZZ-ZD)/Z0))**2/UZZ

            ZZ,UZZ = self.calculateZZZDRV(U10,self.CHMX_nonIrr[c],self.var.HCM[c])
            Z0 = 0.131 * (self.CHMX_nonIrr[c]+0.01) ** 0.997
            ZD = 0.702 * (self.CHMX_nonIrr[c]+0.01) ** 0.979
            RV_nonIrr = 6.25*(np.log((ZZ-ZD)/Z0))**2/UZZ

            VPC1_x,VPC1_y = self.scrp_to_xy(self.var.VPC1[c])
            VPD_slope = (1-VPC1_y) / (VPC1_x-self.var.VPT[c])
            FVPD = np.where(VPD > self.var.VPT[c],np.maximum(0.1,1 - VPD_slope * (VPD - self.var.VPT[c])),1)
            G1 = self.var.GSM[c] * FVPD
            pm1 = self.var.canopy[c]
            # pm1 = 1.
            RC_Irr = np.where(self.var.LAI_Irr_daily[c] > 0,pm1 / ((self.var.LAI_Irr_daily[c]+0.01) * G1 * np.exp(.00155*(330.-self.var.co2))),0)
            RC_nonIrr = np.where(self.var.LAI_nonIrr_daily[c] > 0,pm1 / ((self.var.LAI_nonIrr_daily[c]+0.01) * G1 * np.exp(.00155*(330.-self.var.co2))),0)
            EPP_Irr = np.where(self.var.LAI_Irr_daily[c] > 0,\
                                    pm74 * 1e-3 * (X2 + 86.66 * RHO * VPD / RV_Irr) / (XL * (DLT + GMA * (1. + RC_Irr / RV_Irr))),0) #[m d-1]
            EPP_nonIrr = np.where(self.var.LAI_nonIrr_daily[c] > 0,pm74 * 1e-3 * (X2 + 86.66 * RHO * VPD / RV_nonIrr) / (XL * (DLT + GMA * (1. + RC_nonIrr / RV_nonIrr))),0) #[m d-1]

            if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                self.var.EO_with_Irrcrop[c] = np.where(EO_Irr > EPP_Irr,EO_Irr,EPP_Irr)
            else:
                self.var.EO_with_Irrcrop[c] = np.where(EO_Irrpaddy > EPP_Irr,EO_Irrpaddy,EPP_Irr)
            self.var.EO_with_nonIrrcrop[c] = np.where(EO_nonIrr > EPP_nonIrr,EO_nonIrr,EPP_nonIrr)

            pm41 = self.var.EV_LAI[c]

            CV = 0
            X1 = np.maximum(0.04,np.maximum(.4*self.var.LAI_Irr_daily[c],pm41*(CV+.1)))
            EAJ = np.where(self.var.SnowCover <= 0.005, np.exp(-X1),0.5)
            potential_ES_with_Irrcrop = np.where(self.var.EO_with_Irrcrop[c]>0,self.var.EO_with_Irrcrop[c] * EAJ,0)
            potential_ES_with_Irrcrop = np.where((potential_ES_with_Irrcrop + EPP_Irr)>0,\
                                              np.minimum(potential_ES_with_Irrcrop,potential_ES_with_Irrcrop * self.var.EO_with_Irrcrop[c] / (potential_ES_with_Irrcrop + EPP_Irr)),0)
            potential_ES_with_nonIrrcrop = np.where(self.var.EO_with_nonIrrcrop[c]>0,self.var.EO_with_nonIrrcrop[c] * EAJ,0)
            potential_ES_with_nonIrrcrop = np.where((potential_ES_with_nonIrrcrop + EPP_nonIrr)>0,\
                                              np.minimum(potential_ES_with_nonIrrcrop,potential_ES_with_nonIrrcrop * self.var.EO_with_nonIrrcrop[c] / (potential_ES_with_nonIrrcrop + EPP_nonIrr)),0)

            pm12 = self.var.EV_correct[c]
            pm61 = self.var.EVweight[c]

            XX = potential_ES_with_Irrcrop
            if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                act_ES_with_Irrcrop,SEV_with_Irrcrop = self.calculate_act_ES(XX,pm12,pm61,No=3)
            else:
                act_ES_with_Irrcrop,SEV_with_Irrcrop = self.calculate_act_ES(XX,pm12,pm61,No=2)
            self.var.ES_with_Irrcrop[c] = act_ES_with_Irrcrop
            self.var.ES_with_Irrcrop_layer[c] = SEV_with_Irrcrop

            XX = potential_ES_with_nonIrrcrop
            act_ES_with_nonIrrcrop,SEV_with_nonIrrcrop = self.calculate_act_ES(XX,pm12,pm61,No=1)
            self.var.ES_with_nonIrrcrop[c] = act_ES_with_nonIrrcrop
            self.var.ES_with_nonIrrcrop_layer[c] = SEV_with_nonIrrcrop

            XX = np.maximum(self.var.EO_with_Irrcrop[c] - act_ES_with_Irrcrop,0)
            self.var.EPP_Irr_daily[c] = np.where(EPP_Irr > XX,XX,EPP_Irr)

            XX = np.maximum(self.var.EO_with_nonIrrcrop[c] - act_ES_with_nonIrrcrop,0)
            self.var.EPP_nonIrr_daily[c] = np.where(EPP_nonIrr > XX,XX,EPP_nonIrr)



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
