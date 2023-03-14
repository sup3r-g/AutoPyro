class TR:
    # Transformation Ratio
    column_name = "TR"
    
    def Espitalie_1987(HI, HI_o):
        return (HI_o-HI)*1200/HI_o*(1200-HI) 

    def Peters_2005(HI, HI_o, TOC, TOC_o):
        p = 83.33
        return 1-(HI*TOC*(p-TOC_o))/(HI_o*TOC_o*(p-TOC))


class HIo:
    # Original Hydrogen Index
    column_name = "HIo"
    
    def Cornford_2001(HI, T_max):
        return HI + HI * (T_max - 435) / 30

    def Fit():
        pass


class TOCo:
    # Original Total Organic Carbon
    column_name = "TOCo"
    
    def Peters_2005(p, HI, HI_o, TOC, TR):
        p = 83.33

        return p*HI*TOC/(HI_o*(1 - TR/100)*(p-TOC) + HI*TOC)
    
    def Neruchev_1998():
        pass
