# -------------------------------------------------------------------------
# Name:        Evaporation module
# Purpose:
#
# Author:      PB
#
# Created:     01/08/2016
# Copyright:   (c) PB 2016
# -------------------------------------------------------------------------
"""Modified by Zhonghao Fu, 06/03/2026"""

from laws.management_modules.data_handling import *
import re

class evaporation_crops(object):
    """
    Evaporation module
    Calculate potential evaporation and pot. transpiration


    **Global variables**

    =====================================  ======================================================================  =====
    Variable [self.var]                    Description                                                             Unit
    =====================================  ======================================================================  =====
    snowEvap                               total evaporation from snow for a snow layers                           m
    cropKC_landCover                                                                                               --
    Crops_names                            Internal: List of specific crops                                        --
    activatedCrops                                                                                                 --
    load_initial                           Settings initLoad holds initial conditions for variables                input
    fracCrops_nonIrr                       Fraction of cell currently planted with specific non-irr crops          --
    monthCounter                                                                                                   --
    fracCrops_IrrLandDemand                                                                                        --
    fracCrops_nonIrrLandDemand                                                                                     --
    ratio_a_p_nonIrr                       Ratio actual to potential evapotranspiration, monthly, non-irrigated [  %
    totalPotET_month                                                                                               --
    ratio_a_p_Irr                          Ratio actual to potential evapotranspiration, monthly [crop specific]   %
    Yield_nonIrr                           Relative monthly non-irrigated yield [crop specific]                    %
    currentKY                              Yield sensitivity coefficient [crop specific]                           Posit
    Yield_Irr                              Relative monthly irrigated yield [crop specific]                        %
    currentKC                              Current crop coefficient for specific crops                             --
    generalIrrCrop_max                                                                                             --
    generalnonIrrCrop_max                                                                                          --
    weighted_KC_nonIrr                                                                                             --
    weighted_KC_Irr                                                                                                --
    weighted_KC_Irr_woFallow_fullKc                                                                                --
    _weighted_KC_Irr                                                                                               --
    weighted_KC_Irr_woFallow                                                                                       --
    PotET_crop                                                                                                     --
    totalPotET_month_segment                                                                                       --
    PotETaverage_crop_segments                                                                                     --
    areaCrops_Irr_segment                                                                                          --
    areaCrops_nonIrr_segment                                                                                       --
    areaPaddy_Irr_segment                                                                                          --
    Precipitation_segment                                                                                          --
    availableArableLand_segment                                                                                    --
    cropCorrect                            calibration factor of crop KC factor                                    --
    includeCrops                           1 when includeCrops=True in Settings, 0 otherwise                       bool
    Crops                                  Internal: List of specific crops and Kc/Ky parameters                   --
    potTranspiration                       Potential transpiration (after removing of evaporation)                 m
    cropKC                                 crop coefficient for each of the 4 different land cover types (forest,  --
    minCropKC                              minimum crop factor (default 0.2)                                       --
    irrigatedArea_original                                                                                         --
    frac_totalnonIrr                       Fraction sown with specific non-irrigated crops                         %
    frac_totalIrr_max                      Fraction sown with specific irrigated crops, maximum throughout simula  %
    frac_totalnonIrr_max                   Fraction sown with specific non-irrigated crops, maximum throughout si  %
    GeneralCrop_Irr                        Fraction of irrigated land class sown with generally representative cr  %
    fallowIrr                              Fraction of fallowed irrigated land                                     %
    fallowIrr_max                          Fraction of fallowed irrigated land, maximum throughout simulation      %
    GeneralCrop_nonIrr                     Fraction of grasslands sown with generally representative crop          %
    fallownonIrr                           Fraction of fallowed non-irrigated land                                 %
    fallownonIrr_max                       Fraction of fallowed non-irrigated land, maximum throughout simulation  %
    availableArableLand                    Fraction of land not currently planted with specific crops              %
    cellArea                               Area of cell                                                            m2
    ETRef                                  potential evapotranspiration rate from reference crop                   m
    Precipitation                          Precipitation (input for the model)                                     m
    SnowMelt                               total snow melt from all layers                                         m
    Rain                                   Precipitation less snow                                                 m
    prevSnowCover                          snow cover of previous day (only for water balance)                     m
    SnowCover                              snow cover (sum over all layers)                                        m
    potBareSoilEvap                        potential bare soil evaporation (calculated with minus snow evaporatio  m
    irr_Paddy_month                                                                                                --
    fracCrops_Irr                          Fraction of cell currently planted with specific irrigated crops        %
    actTransTotal_month_nonIrr             Internal variable: Running total of  transpiration for specific non-ir  m
    actTransTotal_month_Irr                Internal variable: Running total of  transpiration for specific irriga  m
    irr_crop_month                                                                                                 --
    frac_totalIrr                          Fraction sown with specific irrigated crops                             %
    weighted_KC_nonIrr_woFallow                                                                                    --
    totalPotET                             Potential evaporation per land use class                                m
    fracVegCover                           Fraction of specific land covers (0=forest, 1=grasslands, etc.)         %
    adminSegments                          Domestic agents                                                         Int
    =====================================  ======================================================================  =====

    **Functions**
    """

    def __init__(self, model):
        """Construct the crop-coupled evaporation module."""
        self.var = model.var
        self.model = model

    def initial(self):
        #no_types = len (self.var.coverTypes)
        self.var.cropKCmonth = np.zeros((4, 13, len(globals.inZero)))
        self.var.cropKC = np.zeros((4, len(globals.inZero)))
        self.var.interceptCap = np.zeros((2, 13, len(globals.inZero)))
        j = 0
        for coverType in self.var.coverTypes:

            if coverType in ['forest', 'grassland', 'irrPaddy', 'irrNonPaddy']:
                for i in range(13):
                    self.var.cropKCmonth[j,i,:] = readnetcdf2(coverType + '_cropCoefficientNC', i*3, "10day")
                    self.var.cropKCmonth[j,i,:] = np.maximum(self.var.cropKCmonth[j,i,:], self.var.minCropKC)
                iii =1

            if coverType in ['forest', 'grassland']:
                for i in range(13):
                    self.var.interceptCap[j,i,:] = readnetcdf2(coverType + '_interceptCapNC', i * 3, "10day")
                    self.var.interceptCap[j,i,:] = np.maximum(self.var.interceptCap[j,i,:], self.var.minInterceptCap[j])
            j = j +1
        ii =1

        self.var.crop_weight_potT = np.tile(globals.inZero, (4, 1))
        self.var.weight_potTranspiration_generalCrop = np.tile(globals.inZero, (4,1))
        self.var.crop_weight_actSEV = np.tile(globals.inZero, (4, 1))
        self.var.crop_weight_actT_soillayer = np.tile(globals.inZero, (4, 3,1))
    def dynamic(self, coverType, No):
        """
        Dynamic part of the soil module

        calculating potential Evaporation for each land cover class with kc factor
        get crop coefficient, use potential ET, calculate potential bare soil evaporation and transpiration

        :param coverType: Land cover type: forest, grassland  ...
        :param No: number of land cover type: forest = 0, grassland = 1 ...
        :return: potential evaporation from bare soil, potential transpiration
        """

        # get crop coefficient
        # to get ETc from ET0 x kc factor  ((see http://www.fao.org/docrep/X0490E/x0490e04.htm#TopOfPage figure 4:)
        # crop coefficient read for forest and grassland from file



        # calculate potential bare soil evaporation - only once
        if No == 0:
            self.var.potBareSoilEvap = self.var.cropCorrect * self.var.minCropKC * self.var.ETRef
            # calculate snow and ice evaporation
            self.var.snowEvap = np.minimum(self.var.SnowMelt, self.var.potBareSoilEvap)
            self.var.potBareSoilEvap -= self.var.snowEvap

            self.var.iceEvap = np.minimum(self.var.IceMelt, self.var.potBareSoilEvap)
            self.var.potBareSoilEvap -= self.var.iceEvap

            self.var.SnowMelt -= self.var.snowEvap
            self.var.IceMelt -= self.var.iceEvap

        #if dateVar['newStart'] or (dateVar['currDate'].day in [1,11,21]):
        #    self.var.cropKC[No] = readnetcdf2(coverType + '_cropCoefficientNC', dateVar['10day'], "10day")
        #    self.var.cropKC[No] = np.maximum(self.var.cropKC[No], self.var.minCropKC)
        #    self.var.cropKC_landCover[No] = self.var.cropKC[No].copy()

        # interpolation for each day from monthly values
        dplus = dateVar['30day'] + 1
        dpart = dateVar['doy'] % 30
        if dplus > 12: dplus = 0
        self.var.cropKC[No] = (self.var.cropKCmonth[No, dplus, :] - self.var.cropKCmonth[No,dateVar['30day'],:]) / 30. * dpart + self.var.cropKCmonth[No,dateVar['30day'],:]
        cropKC_landCover = self.var.cropKC[No]


        if self.var.includeCrops:
            # includeCrops allows for crops and fallow land to makeup the landcovers grasslands and non-paddy, and
            # maintains including a representative vegetation. It is developed to allow users to decide on the crops
            # and parameters that are relevant for the study. The Excel cwatm_settings.xlsx is used to detail the crops
            # and associated parameters. Crops have a unique planting month and four growth stages. Each stage is associated with a
            # crop coefficient (Kc), yield response factor (Ky), and length.

            if No == 1:
                # Only go through this once:
                # I. new start and II. beginning of the month

                # I. new start
                if dateVar['newStart']:

                    for z in ['irrM3_Paddy_month_segment', 'irr_Paddy_month', 'irr_crop', 'irr_crop_month', 'irrM3_crop_month_segment', 'ratio_a_p_nonIrr', 'ratio_a_p_Irr',
                              'fracCrops_IrrLandDemand', 'fracCrops_Irr', 'areaCrops_Irr_segment', 'areaCrops_nonIrr_segment', 'fracCrops_nonIrrLandDemand', 'fracCrops_nonIrr',
                              'activatedCrops', 'monthCounter', 'currentKC', 'totalPotET_month', 'PET_cropIrr_m3',
                              'actTransTotal_month_Irr', 'actTransTotal_month_nonIrr', 'currentKY', 'Yield_Irr',
                              'Yield_nonIrr', 'actTransTotal_crops_Irr', 'actTransTotal_crops_nonIrr', 'PotET_crop', 'PotETaverage_crop_segments', 'totalPotET_month_segment',
                              'ET_crop_nonIrr', 'ET_crop_Irr', 'ratio_a_p_nonIrr_daily', 'ratio_a_p_Irr_daily']:
                        vars(self.var)[z] = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
                    self.var.irr_Paddy_month = globals.inZero.copy()
                    for z in [crop for crop in self.var.Crops_names]:
                        vars(self.var)[z + '_Irr'] = globals.inZero.copy()
                        vars(self.var)[z + '_nonIrr'] = globals.inZero.copy()

                    self.var.fracCrops_Irr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))
                    self.var.fracCrops_nonIrr = np.tile(globals.inZero, (len(self.var.Crops_names), 1))


                if dateVar['newStart'] or dateVar['newYear']:

                    crop_inflate_factor = 1
                    for i in range(len(self.var.Crops_names)):
                        if self.var.crop_coupling:
                            self.var.fracCrops_IrrLandDemand[i] = readnetcdf2(self.var.Crops_names[i] + '_Irr', dateVar['currDate'],
                                                                    'yearly',
                                                                    value='area_frac')
                            self.var.fracCrops_nonIrrLandDemand[i] = readnetcdf2(self.var.Crops_names[i] + '_nonIrr', dateVar['currDate'],
                                                                    'yearly',
                                                                    value='area_frac')

                        else:
                            try:
                                self.var.fracCrops_IrrLandDemand[i] = np.where(
                                    loadmap(self.var.Crops_names[i] + '_Irr') * crop_inflate_factor <= 1,
                                    loadmap(self.var.Crops_names[i] + '_Irr') * crop_inflate_factor, 1)
                                self.var.fracCrops_nonIrrLandDemand[i] = np.where(
                                    loadmap(self.var.Crops_names[i] + '_nonIrr') * crop_inflate_factor <= 1,
                                    loadmap(self.var.Crops_names[i] + '_nonIrr') * crop_inflate_factor,
                                    1)

                            except:

                                self.var.fracCrops_IrrLandDemand[i] = readnetcdf2(self.var.Crops_names[i] + '_Irr', dateVar['currDate'],
                                                                    'yearly',
                                                                    value=re.split(r'[^a-zA-Z0-9_[\]]', cbinding(self.var.Crops_names[i] + '_Irr'))[-2])


                                self.var.fracCrops_nonIrrLandDemand[i] = readnetcdf2(self.var.Crops_names[i] + '_nonIrr', dateVar['currDate'],
                                                                    'yearly',
                                                                    value=re.split(r'[^a-zA-Z0-9_[\]]', cbinding(self.var.Crops_names[i] + '_nonIrr'))[-2])

                        self.var.fracCrops_nonIrrLandDemand[i] = np.maximum(1-0.99998,self.var.fracCrops_nonIrrLandDemand[i])
                        self.var.fracCrops_IrrLandDemand[i] = np.maximum(1-0.99998,self.var.fracCrops_IrrLandDemand[i])
                        # in two places
                        if 'crops_leftoverNotIrrigated' in binding:
                            if i <= int(cbinding('crops_leftoverNotIrrigated')):
                                #print('in evaporation: some crops not rainfed')
                                self.var.fracCrops_nonIrrLandDemand[i] = globals.inZero.copy()

                        # activatedCrops[c] = 1 where crop c is planned in at least 0.001% of the cell, and 0 otherwise.
                        self.var.activatedCrops[i] = np.minimum(np.maximum((self.var.fracCrops_IrrLandDemand[i] +
                                                                            self.var.fracCrops_nonIrrLandDemand[i] + 0.99999) // 1,
                                                                           self.var.activatedCrops[i]), 1)

                if 'moveIrrFallowToNonIrr' in option:
                    if checkOption('moveIrrFallowToNonIrr'):

                        # The irrigated land class may have given up fallow land to the grasslands land class.
                        # If this is the case, these fallow lands are returned to the irrigated land class briefly to
                        # allow them to be planted on in the irrigated land class, and then returned to the
                        # grasslands land class.

                        self.var.fracVegCover[3] = self.var.irrigatedArea_original.copy()

                        remainderLand = np.maximum(
                            globals.inZero.copy() + 1 - self.var.fracVegCover[4] - self.var.fracVegCover[3] -
                            self.var.fracVegCover[5] - self.var.fracVegCover[2] - self.var.fracVegCover[0],
                            globals.inZero.copy())

                        self.var.fracVegCover[1] = remainderLand.copy()


                for c in range(len(self.var.Crops_names)):

                    # Dawn of the next month
                    # We first harvest, and then we plant

                    self.var.GrowPeriod_Irr[c] += np.where(self.var.JJT_Irr[c] > 0, 1, 0)
                    self.var.GrowPeriod_nonIrr[c] += np.where(self.var.JJT_nonIrr[c] > 0, 1, 0)

                    self.var.JJT_Irr[c] = np.where(self.var.JJT_Irr[c] ==3,0,self.var.JJT_Irr[c])
                    self.var.JJT_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] ==3,0,self.var.JJT_nonIrr[c])

                    self.var.GrowPeriod_Irr[c] = np.where(self.var.JJT_Irr[c] > 0, self.var.GrowPeriod_Irr[c], 0)
                    self.var.GrowPeriod_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] > 0, self.var.GrowPeriod_nonIrr[c], 0)


                    self.var.fracCrops_Irr[c] = np.where(self.var.JJT_Irr[c] > 0, self.var.fracCrops_Irr[c], 0)
                    self.var.fracCrops_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] > 0, self.var.fracCrops_nonIrr[c], 0)


                    self.var.actTransForHI_Irr_daily[c] = np.where((self.var.JJT_Irr[c] == 1) | (self.var.JJT_Irr[c] == 2),\
                                                                        self.var.actTransForHI_Irr_daily[c],0)
                    self.var.actTransForHI_nonIrr_daily[c] = np.where((self.var.JJT_nonIrr[c] == 1) | (self.var.JJT_nonIrr[c] == 2),\
                                                                        self.var.actTransForHI_nonIrr_daily[c],0)
                    self.var.potTransForHI_Irr_daily[c] = np.where((self.var.JJT_Irr[c] == 1) |  (self.var.JJT_Irr[c] == 2),\
                                                                        self.var.potTransForHI_Irr_daily[c],0)
                    self.var.potTransForHI_nonIrr_daily[c] = np.where((self.var.JJT_nonIrr[c] == 1) | (self.var.JJT_nonIrr[c] == 2),\
                                                                        self.var.potTransForHI_nonIrr_daily[c],0)

                    # This calculates the current land being used for irrigated and non-irrigated crops
                    frac_totalIrr, frac_totalnonIrr = globals.inZero.copy(), globals.inZero.copy()
                    for i in range(len(self.var.Crops_names)):
                        frac_totalIrr += self.var.fracCrops_Irr[i]
                        frac_totalnonIrr += self.var.fracCrops_nonIrr[i]

                    remainder_land_nonIrr = self.var.fracVegCover[1] - frac_totalnonIrr
                    remainder_land_Irr = self.var.fracVegCover[3] - frac_totalIrr

                    # Sowing seeds, if crop is not already growing, if there is sufficient space
                    # If it is the planting month of the crop,
                    # the crop is planted both irrigated and non-irrigated,
                    # in the remaining available land.
                    self.var.JJT_Irr[c] = np.where((self.var.PD_Irr[c] == dateVar['doy'])& (self.var.JJT_Irr[c] == 0),self.var.activatedCrops[c],self.var.JJT_Irr[c])
                    self.var.JJT_nonIrr[c] = np.where((self.var.PD_nonIrr[c] == dateVar['doy'])& (self.var.JJT_nonIrr[c] == 0),self.var.activatedCrops[c],self.var.JJT_nonIrr[c])
                    # When it is the crop's planting month and it is not yet already planted (the month counter is zero).
                    # The counter only starts if there is some of the crop growing in the cell (it is activated).
                    # Otherwise, the month counter is kept constant
                    self.var.GrowPeriod_Irr[c] = np.where((self.var.PD_Irr[c] == dateVar['doy'])& (self.var.GrowPeriod_Irr[c] == 0), self.var.activatedCrops[c], self.var.GrowPeriod_Irr[c])
                    self.var.GrowPeriod_nonIrr[c] = np.where((self.var.PD_nonIrr[c] == dateVar['doy'])& (self.var.GrowPeriod_nonIrr[c] == 0), self.var.activatedCrops[c], self.var.GrowPeriod_nonIrr[c])

                    self.var.fracCrops_Irr[c] = np.where(self.var.JJT_Irr[c] > 0,
                                                         np.where(remainder_land_Irr > 0,
                                                                  np.minimum(remainder_land_Irr, self.var.fracCrops_IrrLandDemand[c]),
                                                                  0),
                                                         self.var.fracCrops_Irr[c])

                    if 'leftoverIrrigatedCropIsRainfed' in option:
                        if checkOption('leftoverIrrigatedCropIsRainfed'):
                            self.var.fracCrops_nonIrrLandDemand[c] = self.var.fracCrops_IrrLandDemand[c] - \
                                                                        self.var.fracCrops_Irr[c]

                            if 'crops_leftoverNotIrrigated' in binding:
                                if c <= int(cbinding('crops_leftoverNotIrrigated')):
                                    self.var.fracCrops_nonIrrLandDemand[c] = globals.inZero.copy()

                    self.var.fracCrops_nonIrr[c] = np.where(self.var.JJT_nonIrr[c] > 0,
                                                            np.where(remainder_land_nonIrr > 0,
                                                                     np.minimum(remainder_land_nonIrr, self.var.fracCrops_nonIrrLandDemand[c]),
                                                                     0),
                                                            self.var.fracCrops_nonIrr[c])

                    frac_totalIrr, frac_totalnonIrr = globals.inZero.copy(), globals.inZero.copy()
                    for i in range(len(self.var.Crops_names)):
                        frac_totalIrr += self.var.fracCrops_Irr[i]
                        frac_totalnonIrr += self.var.fracCrops_nonIrr[i]

                    # self.var.frac_totalIrr = frac_totalIrr.copy()
                    # self.var.frac_totalnonIrr = frac_totalnonIrr.copy()

                    remainder_land_nonIrr = self.var.fracVegCover[1] - frac_totalnonIrr
                    remainder_land_Irr = self.var.fracVegCover[3] - frac_totalIrr

                frac_totalIrr, frac_totalnonIrr = globals.inZero.copy(), globals.inZero.copy()
                for i in range(len(self.var.Crops_names)):
                    frac_totalIrr += self.var.fracCrops_Irr[i]
                    frac_totalnonIrr += self.var.fracCrops_nonIrr[i]

                self.var.frac_totalIrr = frac_totalIrr.copy()
                self.var.frac_totalnonIrr = frac_totalnonIrr.copy()

                self.var.frac_totalIrr_max = np.maximum(frac_totalIrr, self.var.frac_totalIrr_max)
                self.var.frac_totalnonIrr_max = np.maximum(frac_totalnonIrr, self.var.frac_totalnonIrr_max)
                # UNDER CONSTRUCTION: Automatic fallowing for irrigated land
                self.var.generalIrrCrop_max = np.maximum(self.var.fracVegCover[3] - self.var.frac_totalIrr_max, globals.inZero.copy())
                self.var.generalnonIrrCrop_max = np.maximum(self.var.fracVegCover[1] - self.var.frac_totalnonIrr_max, globals.inZero.copy())
                # The representative vegetation is determined from a specific user-input map, as compared to being
                # determined automatically otherwise.
                if 'GeneralCrop_Irr' in binding and checkOption('use_GeneralCropIrr') == True:
                    self.var.GeneralCrop_Irr = loadmap('GeneralCrop_Irr')
                    self.var.GeneralCrop_Irr = np.minimum(self.var.fracVegCover[3] - frac_totalIrr,
                                                            self.var.GeneralCrop_Irr)

                # Fallowing and general crop are determined automatically, and are not specific input maps.
                elif checkOption('use_GeneralCropIrr') == False:

                    # Fallow land exists alongside general land as non-specific crop options.
                    if checkOption('activate_fallow') == True:

                        # Crop land that has been previously planted by a specific-crop is fallowed between plantings.
                        if checkOption('automaticFallowingIrr') == True:
                            self.var.GeneralCrop_Irr = self.var.generalIrrCrop_max.copy()

                        # With the interest in fallowing without automatic fallowing nor a specific input map implies
                        # the scenario without general lands -- only specific planted crops and fallow land.
                        else:
                            self.var.GeneralCrop_Irr = globals.inZero.copy()

                    else:
                        # activate_fallow = False implies that all non-planted grassland and non-paddy land is made
                        # to be representative vegetation.
                        self.var.GeneralCrop_Irr = self.var.fracVegCover[3] - self.var.frac_totalIrr



                self.var.fallowIrr = self.var.fracVegCover[3] - (self.var.frac_totalIrr + self.var.GeneralCrop_Irr)
                self.var.fallowIrr_max = np.maximum(self.var.fallowIrr, self.var.fallowIrr_max)

                # Updating irrigated land to not include fallow
                # Irrigated fallow land is moved to non-irrigated fallow land. Irrigated fallow land is

                #UNDER CONSTRUCTION
                if 'moveIrrFallowToNonIrr' in option:
                    if checkOption('moveIrrFallowToNonIrr'):

                        self.var.fracVegCover[3] = self.var.frac_totalIrr + self.var.GeneralCrop_Irr
                        remainderLand = np.maximum(
                            globals.inZero.copy() + 1 - self.var.fracVegCover[4] - self.var.fracVegCover[3] -
                            self.var.fracVegCover[5] - self.var.fracVegCover[2] - self.var.fracVegCover[0],
                            globals.inZero.copy())

                        self.var.fracVegCover[1] = remainderLand.copy()


                if 'GeneralCrop_nonIrr' in binding and checkOption('use_GeneralCropnonIrr') == True:

                    self.var.GeneralCrop_nonIrr = loadmap('GeneralCrop_nonIrr')
                    self.var.GeneralCrop_nonIrr = np.minimum(self.var.fracVegCover[1] - frac_totalnonIrr,
                                                                self.var.GeneralCrop_nonIrr)

                elif checkOption('use_GeneralCropnonIrr') == False:
                    if checkOption('activate_fallow') == True:
                        self.var.GeneralCrop_nonIrr = self.var.generalnonIrrCrop_max.copy()
                    else:
                        self.var.GeneralCrop_nonIrr = self.var.fracVegCover[1] - self.var.frac_totalnonIrr

                self.var.fallownonIrr = self.var.fracVegCover[1] - (
                        self.var.frac_totalnonIrr + self.var.GeneralCrop_nonIrr)
                self.var.fallownonIrr_max = np.maximum(self.var.fallownonIrr, self.var.fallownonIrr_max)

                self.var.availableArableLand = self.var.fallowIrr + self.var.fracVegCover[1] - frac_totalnonIrr

            if No == 1:

                ## only once##
                self.var.weighted_actT_soillayer_Irr = np.tile(globals.inZero, (3, 1))
                self.var.weighted_actT_soillayer_nonIrr = np.tile(globals.inZero, (3, 1))
                self.var.weight_actSEV_soillayer = np.tile(globals.inZero, (4, 2,1))

                availWaterPlant1 = self.var.w1[No] - self.var.wwp1[No] # * self.var.rootDepth[0][No]  should not be multiplied again with soildepth
                availWaterPlant2 = self.var.w2[No] - self.var.wwp2[No]  # * self.var.rootDepth[1][No]
                availWaterPlant3 = self.var.w3[No] - self.var.wwp3[No]  # * self.var.rootDepth[1][No]
                # availWaterPlant3 = np.maximum(0., self.var.w3[No] - self.var.wwp3[No])  #* self.var.rootDepth[2][No]
                readAvlWater = availWaterPlant1 + availWaterPlant2 + availWaterPlant3 # + availWaterPlant3

                # Retain the CWatM weighting method for calculating
                # evapotranspiration in each grid cell.
                # Calculate weighted potential transpiration.
                self.var.weighted_potT_nonIrr = self.var.GeneralCrop_nonIrr * np.maximum(0., cropKC_landCover* self.var.ETRef - self.var.potBareSoilEvap)
                for c in range(len(self.var.Crops_names)):
                    self.var.weighted_potT_nonIrr += self.var.fracCrops_nonIrr[c] * self.var.EPP_nonIrr_daily[c]
                self.var.weighted_potT_nonIrr = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weighted_potT_nonIrr / self.var.fracVegCover[No], 0)
                self.var.crop_weight_potT[No] = self.var.weighted_potT_nonIrr.copy()

                self.var.crop_weight_potT[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_potT[No])
                #######
                # Weight actual transpiration by soil layer. Specific crops are
                # calculated here; general crops are handled by the soil module.
                self.var.weight_potTranspiration_generalCrop[No] = np.where(self.var.fracVegCover[No] > 0,
                    self.var.GeneralCrop_nonIrr * np.maximum(0., cropKC_landCover* self.var.ETRef - self.var.potBareSoilEvap) / self.var.fracVegCover[No],0)
                RTO = np.where(self.var.totAvlWater_grassland > 0, readAvlWater/self.var.totAvlWater_grassland, 0)
                rws = np.where((RTO > 0) & (RTO < self.var.GernalCropWaternonDeficit), RTO / (RTO + np.exp(self.var.GernalCropWaterDeficit_1 - self.var.GernalCropWaterDeficit_2 * RTO)), 1.)
                self.var.rws_nonIrr = rws


                self.var.weighted_actT_soillayer_nonIrr[1] = self.var.GeneralCrop_nonIrr * np.maximum(0., cropKC_landCover* self.var.ETRef - self.var.potBareSoilEvap) * rws
                for c in range(len(self.var.Crops_names)):
                    self.var.weighted_actT_soillayer_nonIrr += self.var.fracCrops_nonIrr[c] * self.var.act_Transpt_soilLayer_nonIrr_daily[c]
                self.var.weighted_actT_soillayer_nonIrr = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weighted_actT_soillayer_nonIrr / self.var.fracVegCover[No], 0)
                self.var.crop_weight_actT_soillayer[No] = self.var.weighted_actT_soillayer_nonIrr.copy()
                self.var.crop_weight_actT_soillayer[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_actT_soillayer[No])
                self.var.weight_crop_actT[No] = np.sum(self.var.crop_weight_actT_soillayer[No],axis=0)

                ###Weight actual soil evaporation in different soil layers
                self.var.weight_actSEV_soillayer[No] = self.var.GeneralCrop_nonIrr * self.var.ES_without_nonIrrcrop_layer
                for c in range(len(self.var.Crops_names)):
                    self.var.weight_actSEV_soillayer[No] += self.var.fracCrops_nonIrr[c] * self.var.ES_with_nonIrrcrop_layer[c]
                self.var.weight_actSEV_soillayer[No] += self.var.fallownonIrr * self.var.ES_without_nonIrrcrop_layer
                self.var.weight_actSEV_soillayer[No] = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weight_actSEV_soillayer[No] / self.var.fracVegCover[No], 0)
                self.var.crop_weight_actSEV[No] = np.sum(self.var.weight_actSEV_soillayer[No],axis=0)

                self.var.crop_weight_actSEV[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_actSEV[No])

            if No == 3:
                availWaterPlant1 = self.var.w1[No] - self.var.wwp1[No] # * self.var.rootDepth[0][No]  should not be multiplied again with soildepth
                availWaterPlant2 = self.var.w2[No] - self.var.wwp2[No]  # * self.var.rootDepth[1][No]
                # availWaterPlant3 = np.maximum(0., self.var.w3[No] - self.var.wwp3[No])  #* self.var.rootDepth[2][No]
                readAvlWater = availWaterPlant1 + availWaterPlant2  # + availWaterPlant3

                self.var.weighted_potT_Irr = self.var.GeneralCrop_Irr * np.maximum(0., cropKC_landCover* self.var.ETRef - self.var.potBareSoilEvap)
                for c in range(len(self.var.Crops_names)):
                    if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                        self.var.weighted_potT_Irr += self.var.fracCrops_Irr[c] * self.var.EPP_Irr_daily[c]
                self.var.weighted_potT_Irr = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weighted_potT_Irr / self.var.fracVegCover[No], 0)
                self.var.crop_weight_potT[No] = self.var.weighted_potT_Irr.copy()

                self.var.crop_weight_potT[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_potT[No])

                ###Weight actual plant transpiration in different soil layers，
                # crop-specific PT is calculated first，general crop is calculated in soil module##
                RTO = np.where(self.var.totAvlWater > 0, readAvlWater/self.var.totAvlWater, 0)
                rws = np.where((RTO > 0) & (RTO < self.var.GernalCropWaternonDeficit), RTO / (RTO + np.exp(self.var.GernalCropWaterDeficit_1 - self.var.GernalCropWaterDeficit_2 * RTO)), 1.)
                self.var.rws_Irr = rws
                self.var.weighted_actT_soillayer_Irr[1] = self.var.GeneralCrop_Irr * np.maximum(0., cropKC_landCover* self.var.ETRef - self.var.potBareSoilEvap) * rws
                for c in range(len(self.var.Crops_names)):
                    if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                        self.var.weighted_actT_soillayer_Irr += self.var.fracCrops_Irr[c] * self.var.act_Transpt_soilLayer_Irr_daily[c]

                self.var.weighted_actT_soillayer_Irr = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weighted_actT_soillayer_Irr / self.var.fracVegCover[No], 0)

                self.var.crop_weight_actT_soillayer[No] = self.var.weighted_actT_soillayer_Irr.copy()
                self.var.crop_weight_actT_soillayer[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_actT_soillayer[No])
                self.var.weight_crop_actT[No] = np.sum(self.var.crop_weight_actT_soillayer[No],axis=0)

                ###Weight actual soil evaporation in different soil layers
                self.var.weight_actSEV_soillayer[No] = self.var.GeneralCrop_Irr * self.var.ES_without_Irrcrop_layer

                for c in range(len(self.var.Crops_names)):
                    if self.var.Crops_names[c] != 'Rice1' and self.var.Crops_names[c] != 'Rice2':
                        self.var.weight_actSEV_soillayer[No] += self.var.fracCrops_Irr[c] * self.var.ES_with_Irrcrop_layer[c]

                self.var.weight_actSEV_soillayer[No] += self.var.fallowIrr * self.var.ES_without_crop_layer

                self.var.weight_actSEV_soillayer[No] = np.where(self.var.fracVegCover[No] > 0,
                                                    self.var.weight_actSEV_soillayer[No] / self.var.fracVegCover[No], 0)

                self.var.crop_weight_actSEV[No] = np.sum(self.var.weight_actSEV_soillayer[No],axis=0)
                self.var.crop_weight_actSEV[No] = np.where(self.var.FrostIndex > self.var.FrostIndexThreshold, 0., self.var.crop_weight_actSEV[No])




        ## potTranspiration: Transpiration for each land cover class
        if No == 0 or No == 2:
            self.var.potTranspiration[No] = np.maximum(0., \
                                                       self.var.cropCorrect * self.var.crop_correct_landCover[No] * self.var.cropKC[No] * self.var.ETRef - self.var.potBareSoilEvap) #Dealt with above - self.var.snowEvap)
        else:
            self.var.potTranspiration[No] = self.var.crop_correct_landCover[No] * self.var.crop_weight_potT[No]





        if checkOption('calcWaterBalance'):
            self.model.waterbalance_module.waterBalanceCheck(
                [self.var.Rain,self.var.Snow],  # In
                [self.var.Rain,self.var.SnowMelt,self.var.IceMelt,self.var.snowEvap,self.var.iceEvap],  # Out
                [self.var.prevSnowCover],   # prev storage
                [self.var.SnowCover],
                "Snow2", False)
