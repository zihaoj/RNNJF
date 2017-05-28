#import ROOT
import AtlasStyle as Atlas
from math import sqrt, log
from optparse import OptionParser
import numpy as np
from LaurenColor import *
from array import array
from copy import deepcopy


def GetPlot(infile, dir, name):
    hcf = infile.Get( dir + name)
    return hcf

def AddHistPairs(Hist1, Hist2):
    newList = []
    for i in range(len(Hist1)):
        if Hist1[i] == None:
            newList.append(None)
            continue
        newHist = CopyHist(Hist1[i])
        newHist.Add(Hist2[i])
        newList.append(newHist)
    return newList


def ResetBins(HistList, bin1, bin2):
    for i in range(len(HistList)):
        if HistList[i] == None:
            continue
        for ibin in range(HistList[i].GetNbinsX()+2):
            if ibin< bin1 or ibin>bin2:
                HistList[i].SetBinContent(ibin, 0)
                HistList[i].SetBinError(ibin, 0)
            
def GetHistMedian(Hist):
    nBins = Hist.GetXaxis().GetNbins()
    x = array('d')
    y = array('d')
    for i in range(nBins):
        x.append(Hist.GetBinCenter(i))
        y.append(Hist.GetBinContent(i))
    median = ROOT.TMath.Median(nBins, x, y)
    return median

def GetGausMean(Hist):
    Hist.Fit("gaus", "Q", "")
    gaus_initial = Hist.GetFunction("gaus")
    last_mean = gaus_initial.GetParameter(1)
    last_mean_error = gaus_initial.GetParError(1)
    last_sigma = gaus_initial.GetParameter(2)
    for i in range(20):
        #low = Hist.FindBin(last_mean- 2*last_sigma)        
        #up = Hist.FindBin(last_mean + 2*last_sigma)
        low = last_mean- 2*last_sigma
        up = last_mean + 2*last_sigma

        Hist.Fit("gaus", "Q", "", low, up)
        this_gaus = Hist.GetFunction("gaus")
        this_mean = this_gaus.GetParameter(1)
        this_mean_error = this_gaus.GetParError(1)
        this_sigma = this_gaus.GetParameter(2)
        if abs(this_mean-last_mean)<0.02:
            return this_mean, this_mean_error
        else:
            last_mean = this_mean
            last_mean_error = this_mean_error
            last_sigma = this_sigma

    return this_mean, this_mean_error


def DrawGausMean(Hist):
    Hist.Fit("gaus", "Q", "")
    gaus_initial = Hist.GetFunction("gaus")
    last_mean = gaus_initial.GetParameter(1)
    last_mean_error = gaus_initial.GetParError(1)
    last_sigma = gaus_initial.GetParameter(2)
    for i in range(20):
        #low = Hist.FindBin(last_mean- 2*last_sigma)        
        #up = Hist.FindBin(last_mean + 2*last_sigma)
        low = last_mean- 2*last_sigma
        up = last_mean + 2*last_sigma

        Hist.Fit("gaus", "Q", "", low, up)
        this_gaus = Hist.GetFunction("gaus")
        this_mean = this_gaus.GetParameter(1)
        this_mean_error = this_gaus.GetParError(1)
        this_sigma = this_gaus.GetParameter(2)
        if abs(this_mean-last_mean)<0.02:
            return this_mean, this_mean_error
        else:
            last_mean = this_mean
            last_mean_error = this_mean_error
            last_sigma = this_sigma

    return this_mean, this_mean_error


def CalSysVariation(Central, Variations, precal = False):
    NBins = Central.GetNbinsX()

    if precal:
        for bin in range(NBins):
            origuncert = Central.GetBinError(bin+1)
            totaluncert = origuncert**2
            for j in range(len(Variations)):
                thisuncert = abs(Variations[j].GetBinContent(bin+1)-1)
                totaluncert += thisuncert**2
            totaluncert = sqrt(totaluncert)
            print (totaluncert)
            Central.SetBinError(bin+1, totaluncert)
        return

    for bin in range(NBins):
        origuncert = Central.GetBinError(bin+1)
        totaluncert = origuncert**2
        for j in range(len(Variations)):
            thisuncert = abs(Variations[j].GetBinContent(bin+1)-Central.GetBinContent(bin+1))
            totaluncert += thisuncert**2
        totaluncert = sqrt(totaluncert)
        print (totaluncert)
        Central.SetBinError(bin+1, totaluncert)


def CalSysDiff(Central, Variations):

    diff = CopyHist(Central, True)
    NBins = Central.GetNbinsX()    
    for bin in range(NBins):
        thisuncert = abs(Variations.GetBinContent(bin+1)-Central.GetBinContent(bin+1))
        diff.SetBinContent(bin+1, 1)
        diff.SetBinError(bin+1, thisuncert)

    diff.SetFillStyle(3004)
    diff.SetFillColor(1)
    diff.SetMarkerStyle(10)
    diff.SetMarkerSize(0)

    return diff

def GetRatios(HistsNum, HistsDenom):
    Ratios = []
    for i in range(len(HistsDenom)):

        num = CopyHist(HistsNum[i])
        denom = CopyHist(HistsDenom[i])
        num.Divide(denom)
        thisRatio = CopyHist(num)
        Ratios.append(thisRatio)
    return Ratios

def GetAveRatios(Hists1, Hists2):
    newhists = []
    for i in range(len(Hists1)):

        hist1 = CopyHist(Hists1[i])
        hist2 = CopyHist(Hists2[i])
        newhist = CopyHist(Hists1[i], True)
        for ibin in range(hist1.GetNbinsX()):
            newhist.SetBinContent(ibin+1, (abs(hist1.GetBinContent(ibin+1)-1)+1+abs(hist2.GetBinContent(ibin+1)-1)+1 )/2 )

        newhists.append(newhist)
    return newhists


def GetHistIQR(Hist):
    nBins = Hist.GetXaxis().GetNbins()
    x = []
    total = Hist.Integral()
    check_q25 = True
    check_q75 = True

    for i in range(nBins):
        thisbin = Hist.Integral(0,i+1)
        if check_q25 and thisbin > total/4.:
            q25 = Hist.GetBinCenter(i+1)
            check_q25 = False
        if check_q75 and thisbin > total/4.*3.:
            q75 = Hist.GetBinCenter(i+1)
            check_q75 = False
            
    IQR = q75 - q25
#    q75, q25 = np.percentile(x, [75 ,25])
    return IQR

def CopyHist(hist, doempty = False):
    if hist ==None:
        return None

    nbinsx = hist.GetNbinsX()
    upx = hist.GetXaxis().GetBinUpEdge(nbinsx)
    lowx = hist.GetXaxis().GetBinLowEdge(1)
    title = hist.GetTitle()
    if hist.ClassName() == "TH1D" or "TH1F":
        newhist = ROOT.TH1D(title + ' copy', title + ' copy', nbinsx, lowx, upx)
        hist.Copy(newhist)
        
        if doempty:
            for i in range(nbinsx+1):
                newhist.SetBinContent(i+1, 0)
                newhist.SetBinError(i+1, 0)
                newhist.SetEntries(0)
        return newhist

    if hist.ClassName() == "TH2D" or "TH2F":
        nbinsy = hist.GetNbinsY()
        upy = hist.GetYaxis().GetBinUpEdge(nbinsy)
        lowy = hist.GetYaxis().GetBinLowEdge(1)

        newhist = ROOT.TH2D(title + ' copy', title + ' copy', nbinsx, lowx, upx, nbinsy, lowy, upy)
        hist.Copy(newhist)
        if doempty:
            for i in range(nbinsx+1):
                for j in range(nbinsy+1):
                    newhist.SetBinContent(i+1, j+1, 0)
                    newhist.SetBinError(i+1, j+1, 0)
                    newhist.SetEntries(0)
        return newhist

    if hist.ClassName() == "TH3D":
        nbinsy = hist.GetNbinsY()
        upy = hist.GetYaxis().GetBinUpEdge(nbinsy)
        lowy = hist.GetYaxis().GetBinLowEdge(1)

        nbinsz = hist.GetNbinsZ()
        upz = hist.GetZaxis().GetBinUpEdge(nbinsz)
        lowz = hist.GetZaxis().GetBinLowEdge(1)

        newhist = ROOT.TH3D(title + ' copy', title + ' copy', nbinsx, lowx, upx, nbinsy, lowy, upy, nbinsz, lowz, upz)
        hist.Copy(newhist)
        if doempty:
            for i in range(nbinsx+1):
                for j in range(nbinsy+1):
                    for k in range(nbinsz+1):
                        newhist.SetBinContent(i+1, j+1, k+1, 0)
                        newhist.SetBinError(i+1, j+1, k+1, 0)
                        newhist.SetEntries(0)
        return newhist
    return

def CopyHists(listhist, doempty = False):
    outlist = []
    for i in range(len(listhist)):
        thisnew = CopyHist(listhist[i], doempty)
        outlist.append(thisnew)
    return outlist

def NormHists(hists):
    newlist = []
    for i in range(len(hists)):
        if hists[i] != None and hists[i].Integral() !=0 :

            newhist = CopyHist(hists[i])
            newhist.Scale(1/newhist.Integral())
            newlist.append(newhist)
        else:
            newlist.append(None)
    return newlist
        

def ScaleHist(hist, bins, factors):
    newhist = CopyHist(hist)
    if len(bins) != len(factors):
        print ('ERROR: bins and factors length not the same')
        return

    for i in range(len(bins)):
        thisval = newhist.GetBinContent(bins[i])
        thiserr = newhist.GetBinError(bins[i])
        thisval = thisval/factors[i]

        thiserr = thiserr/factors[i]
        newhist.SetBinContent(bins[i], thisval)
        newhist.SetBinError(bins[i], thiserr)
    return newhist

def StackHists(inplots, inlabel):
    outplots = []
    outlabel = []
    count =0
    aggsum = 0

    for i in range(len(inplots)):
        count += 1
        if inplots[i] == None:
            outplots.append(None)
            outlabel.append("no this plot")
        else:
            outplots.append(CopyHist(inplots[i]))
            outlabel.append(inlabel[i])
            aggsum = CopyHist(inplots[i])
            break

    for i in range(count, len(inplots)):
        if inplots[i] == None:
            outplots.append(None)
            outlabel.append("no this plot")
        else:
            outlabel.append(inlabel[i])
            newhist = CopyHist(inplots[i])
            aggsum.Add(newhist)
            newagg = CopyHist(aggsum)
            outplots.append(newagg)

    sumplot = None
    outplots.reverse()
    outlabel.reverse()
    for i in range(len(outplots)):
        if outplots[i] != None:
            sumplot = CopyHist(outplots[i])
            break

    ##outplots = [sumplot]+outplots
    #outlabel = ["dummy"]+outlabel

    outplots.append(sumplot)
    outlabel.append("MC Uncertainty")
    copysumplot = CopyHist(sumplot)
    return outplots, outlabel, copysumplot

def project1D(hist, bin1, bin2, axis):
    if axis == "x" or axis =="X":
        projection = hist.ProjectionX(hist.GetName()+"_px", bin1, bin2)
    if axis == "y" or axis =="Y":
        projection = hist.ProjectionY(hist.GetName()+"_py", bin1, bin2)
    return projection

def Project2D(inplot, dn, up):
    if "TH3" not in inplot.ClassName():
        return 
    nbinsx = inplot.GetNbinsX()
    nbinsy = inplot.GetNbinsY()
    nbinsZ = inplot.GetNbinsZ()

    upx = inplot.GetXaxis().GetBinUpEdge(nbinsx)
    lowx = inplot.GetXaxis().GetBinLowEdge(1)

    upy = inplot.GetYaxis().GetBinUpEdge(nbinsy)
    lowy = inplot.GetYaxis().GetBinLowEdge(1)

    newTH2 = ROOT.TH2D(inplot.GetName()+str(dn)+"_"+str(up), inplot.GetName()+str(dn)+"_"+str(up), nbinsx, lowx, upx, nbinsy, lowy, upy)


    for j in range(nbinsx):
        for k in range(nbinsy):
            ptsum = 0
            pterr = 0
            for i in range(dn, up+1):
                ptsum += inplot.GetBinContent(j+1, k+1, i)
                pterr += inplot.GetBinError(j+1, k+1, i)

            newTH2.SetBinContent(j+1, k+1, ptsum)
            newTH2.SetBinError(j+1, k+1, sqrt(pterr))
    return newTH2
        

class HistoTool:

    def __init__(self, drawratio = "Data / MC", studytype = "Internal", lumi = "20.3", sqrtS = "8", doAtlasLabel = True, doLabel = True, doRescale = True, doLogX = False, doLogY = False, doLogZ = False):
        self.doLabel = doLabel
        self.doAtlasLabel = doAtlasLabel
        self.AtlasLabelPos = 0.2
        self.studytype = studytype
        self.lumi = lumi
        self.sqrtS = sqrtS
        self.doRescale = doRescale
        self.doLogX = doLogX
        self.doLogY = doLogY
        self.doLogZ = doLogZ
        self.texts = []
        self.legend = [0.7, 0.75, 0.93, 0.93]
        self.colorlist = [2, 4, 8, 28, 51, 93, 30, 38, 41, 42, 46]
        self.colorlist = colorind#self.SetPublishableColor()
        self.origcolorlist = colorind#self.SetPublishableColor()
        self.drawOption = ""
        self.FillStyle = 3008
        self.CompareData = False
        self.MCscale = 1.0
        self.DrawScale = 1.0
        self.DrawRatio = drawratio
        self.doPrintPlots = False
        self.OutPlotDir = ""
        self.shift = 0.13
        self.doDiff = False

    def SetPublishableColor(self):
        colors =[(150,45,62),(52,54,66),(151,156,156),(52,136,153),(51,37,50),(100,77,82),(247,122,82),(255,151,79),(164,154,135), (242,235,199)]
        tcolors=[]
        colorind=[]
        colorIndBase = 2000;
        for i,c in enumerate(colors):
        #tcolors.append(ROOT.TColor(colorIndBase+i,float(c[0])/255.0,float(c[1])/255.0,float(c[2])/255.0))
            tcolors.append(ROOT.TColor(colorIndBase+i,float(c[0])/255.0,float(c[1])/255.0,float(c[2])/255.0))
            colorind.append(colorIndBase+i)

        print (tcolors, colorind)
        return colorind

    def SetDrawOption(self, option):
        self.drawOption = option

    def SetMCscale(self, scale):
        self.MCscale = scale

    def SetDrawScale(self, scale):
        self.DrawScale = scale

    def SetFillStyle(self, option):
        self.FillStyle = option

    def SetCompData(self, do):
        self.CompareData = do

    def ClearDrawOption(self):
        self.drawOption = ""

    def SetColors(self, colorlist):
        self.colorlist = colorlist

    def ReSetColors(self):
        self.colorlist = self.origcolorlist

    def SetLumi(self, lumi):
        self.lumi = lumi

    def SetSqrtS(self, SqrtS):
        self.sqrtS = SqrtS

    def AddTexts(self, x, y, color=1, size =0.08, text=""):
        self.texts.append([x, y, color, size, text])

    def ClearTexts(self):
        self.texts = []

    def DoLogX(self):
        self.doLogX = True

    def UndoLogX(self):
        self.doLogX = False

    def DoRescale(self):
        self.doRescale = True

    def UndoRescale(self):
        self.doRescale = False

    def DoLogY(self):
        self.doLogY = True

    def UndoLogY(self):
        self.doLogY = False

    def DoLogZ(self):
        self.doLogZ = True

    def UndoLogZ(self):
        self.doLogZ = False
    
    def DoPrintPlots(self):
        self.doPrintPlots = True

    def UndoPrintPlots(self):
        self.doPrintPlots = False

    def SetOutPlotDir(self, Dir):
        self.OutPlotDir = Dir

    def getROC( self, signal, background, label, cut_start=None, cut_end=None):
        
        ROCList = []
        for ivar in range(len(signal)):
            s_sort = np.sort( signal[ivar] )
            b_sort = np.sort( background[ivar] )

            #c_start=(0.0 if cut_start==None else cut_start)
            #c_end=  (1.0 if cut_end==None else cut_end)
		
            print (s_sort, b_sort)

            for i in range(s_sort.shape[0]):
                if s_sort[i] == float("Inf"):
                    s_sort[i] = 100000
                if s_sort[i] == float("-Inf"):
                    s_sort[i] = -1000000

            for i in range(b_sort.shape[0]):
                if b_sort[i] == float("Inf"):
                    b_sort[i] = 100000
                if b_sort[i] == float("-Inf"):
                    b_sort[i] = -1000000

            c_start=np.min( (s_sort[0], b_sort[0]) )
            c_end=  np.max( (s_sort[len(s_sort)-1], b_sort[len(b_sort)-1]) )

            if c_start==-float('inf'):
                c_start = -2*c_end

            print (label[ivar], "min(", s_sort[0],  b_sort[0],  ")=", c_start)
            print (label[ivar], "max(", s_sort[-1], b_sort[-1], ")=", c_end)

            s_eff=[]
            b_rej=[]

            n_points = 1000
            c_delta = (1.0*c_end - 1.0*c_start) / (1.0*n_points)
            for i in range(1000):
                cut = c_start + i*1.0*c_delta
                s_eff.append( 1.0*np.count_nonzero( s_sort > cut ) / (1.0*len(s_sort))  )
                b_count = np.count_nonzero( b_sort > cut )
                b_rej.append(  (1.0*len(b_sort)) / (1.0 if b_count==0 else (1.0*b_count))  )

            ROC = ROOT.TGraph(n_points, array('d', s_eff), array('d', b_rej))
            ROC.SetName("ROC_%i" % (ivar))

            ROCList.append(ROC)

        canvas = ROOT.TCanvas("ROC_Overlay", "ROC_Overlay", 800, 600)
        canvas.cd()
        mg = ROOT.TMultiGraph()
        legend = ROOT.TLegend(0.5, 0.5, 0.75, 0.75)
        for i in range(len(ROCList)):
            ROC = ROCList[i]
            ROC.SetLineWidth(2)
            ROC.SetLineColor(self.colorlist[i])
            ROC.SetMarkerColor(self.colorlist[i])
            mg.Add(ROC)
            legend.AddEntry(ROC, label[i], "lp")

        mg.Draw("AL")
        mg.GetXaxis().SetTitle("signal efficiency")
        mg.GetYaxis().SetTitle("background rejection")
        legend.Draw()
        canvas.Write()
        canvas.Close()


    def DrawHists(self, title, axisname=[], inplots =[], inlabel=[], instacks=[], instacklabel = [], sys = []):
        maxval =1
        minval =1000
        secminval = 10000
        legend = ROOT.TLegend(self.legend[0], self.legend[1], self.legend[2], self.legend[3])
        legend.SetFillColor(0)
        doExist = True


        ## finding the right axises space         
        for i in range(len(inplots)):
            if inplots[i] == None:
                continue
            doExist = False

            if ("TH2D" in inplots[i].ClassName()):
                continue
                
            if ("TH" not in inplots[i].ClassName()):
                legend.AddEntry(inplots[i], inlabel[i], "LPS")
                continue

            inplots[i] = CopyHist(inplots[i])
            legend.AddEntry(inplots[i], inlabel[i], "LPS")
            thismax = inplots[i].GetMaximum()
            thismin = inplots[i].GetMinimum()

            if maxval< thismax:
                maxval = thismax
            if (minval >= thismin):
                minval = thismin

            inplots[i].GetYaxis().SetTitleOffset( inplots[i].GetYaxis().GetTitleOffset()*1.1)

        for i in range(len(instacks)):
            ## scale the MC
            if instacks[i] == None:
                continue

            doExist = False

            instacks[i] = CopyHist(instacks[i])
            if ((i != len(instacks)-1) and self.CompareData):
                instacks[i].Scale(self.DrawScale)
                legend.AddEntry(instacks[i], instacklabel[i], 'f')

            if ((i == len(instacks)-1) and self.CompareData):
                legend.AddEntry(instacks[i], instacklabel[i])

            instacks[i].GetYaxis().SetTitleOffset( instacks[i].GetYaxis().GetTitleOffset()*1.1)

            thismax = instacks[i].GetMaximum()
            thismin = instacks[i].GetMinimum()

            if maxval< thismax:
                maxval = thismax
            if (minval >= thismin and thismin != 0):
                minval = thismin

        if doExist:
            return 

        if minval <= 1.0:
            minval = 1.0

        ###### draw histogram
        Canv = ROOT.TCanvas('Canv_' + title, 'Canv_' + title, 0, 0, 800, 600)
        if(self.CompareData):
            Pad1 = ROOT.TPad('Pad1', 'Pad1', 0.0, 0.25, 1.0, 0.99, 0)
            Pad2 = ROOT.TPad('Pad2', 'Pad2', 0.0, 0.00, 1.0, 0.32, 0)
            Pad2.SetBottomMargin(0.4)
            Pad1.Draw()
            Pad2.Draw();
            Pad1.cd()

        ncolor =0

        for i in range(len(instacks)):
            if instacks[i]==None:
                ncolor+=1            
                continue
            instacks[i].SetMarkerColor(self.colorlist[ncolor])
            instacks[i].SetFillColor(self.colorlist[ncolor])
            instacks[i].SetFillStyle(self.FillStyle)
            instacks[i].SetLineColor(self.colorlist[ncolor])
            instacks[i].SetLineColor(1)
            instacks[i].SetLineWidth(1)
            instacks[i].GetXaxis().SetTitle(axisname[0])
            instacks[i].GetYaxis().SetTitle(axisname[1])

            if (self.doRescale and not(self.doLogY)):
                instacks[i].GetYaxis().SetRangeUser(0, maxval*4.5/3.)
            if (self.doRescale and self.doLogY):
                instacks[i].GetYaxis().SetRangeUser(minval/100., maxval*10.)
            ncolor+=1            

        for i in range(len(inplots)):
            if self.CompareData:
                XaxisTitle = inplots[i].GetXaxis().GetTitle()
                labelsize = inplots[i].GetXaxis().GetLabelSize()
                inplots[i].SetTitle("")
                inplots[i].GetXaxis().SetLabelSize(0)

                if  i == len(inplots)-1:
                    if ("TH" not in inplots[i].ClassName()):
                        inplots[i].Draw("")
                    else:
                        inplots[i].Draw("e")
                    inplots[i].SetLineColor(1)
                    inplots[i].SetMarkerColor(1)
                    #inplots[i].SetFillColor(1)
                    continue

                if  i == len(inplots)-2:
                    if ("TH" not in inplots[i].ClassName()):
                        inplots[i].Draw("same")
                    else:
                        inplots[i].Draw("e same")
                    inplots[i].SetMarkerStyle(20)
                    inplots[i].SetMarkerColor(2)
                    inplots[i].SetLineColor(2)

                    #### pay attention ##

                    Pad2.cd()
                    relsize = Pad2.GetAbsHNDC()/Pad1.GetAbsHNDC()
                    size = Atlas.tsize/relsize

                    Ratio = None

                    if self.doDiff:
                        Fit = None
                        if ("TH" not in inplots[i].ClassName()):
                            Ratio = deepcopy (inplots[len(inplots)-1])
                            PreBinInt = Ratio.Integral()
                            Fit = deepcopy(inplots[i])

                        else:
                            Fit = deepcopy(inplots[len(inplots)-1])
                            Ratio = deepcopy (inplots[i])
                            PreBinInt = Ratio.Integral()



                        data = deepcopy(Ratio)
                        Ratio.Add(Fit, -1)
                        Ratio.Rebin(5)
                        data.Rebin(5)
                        Ratio.Divide(data)
                        try:
                            Fit.Rebin(5)
                        except AttributeError:
                            None
                    else:

                        Ratio = deepcopy (inplots[len(inplots)-1])
                        Ratio.Divide( inplots[i])

                    Ratio.SetTitle("")
                    Ratio.GetXaxis().SetLabelSize(size)
                    Ratio.GetYaxis().SetLabelSize(size)
                    Ratio.GetXaxis().SetTitleSize(size)
                    Ratio.GetYaxis().SetTitleSize(size) 
                    Ratio.GetXaxis().SetTitleOffset( Ratio.GetXaxis().GetTitleOffset()*relsize*2.9)
                    Ratio.GetXaxis().SetLabelOffset(0.03)
                    Ratio.GetYaxis().SetTitleOffset( Ratio.GetYaxis().GetTitleOffset()*relsize)
                    Ratio.GetYaxis().SetTitle(self.DrawRatio)
                    Ratio.GetXaxis().SetTitle(XaxisTitle)
                    Ratio.GetYaxis().SetNdivisions(4)
                    if self.doDiff:
                        Ratio.GetYaxis().SetRangeUser(-0.2, 0.2)
                    else:
                        Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
                    
                    Ratio.SetMarkerColor(1)
                    Ratio.SetLineColor(1)
                    Ratio.GetYaxis().SetNdivisions(5, ROOT.kFALSE)
                    if sys ==[]:
                        Ratio.Draw('e')
                    else:
                        Ratio.Draw('e')
                        sys[0].SetFillStyle(3004)
                        sys[0].SetFillColor(1)
                        sys[0].SetMarkerStyle(10)
                        sys[0].SetMarkerSize(0)

                        if sys[0].GetBinError(1)!= 0:
                            sys[0].Draw("e2 same")
                        if len(sys)>1:
                            sys[1].SetFillStyle(3004)
                            sys[1].SetFillColor(2)
                            sys[1].SetMarkerStyle(10)
                            sys[1].SetMarkerSize(0)
                            sys[1].SetLineColor(2)
                            sys[1].SetLineWidth(2)
                            sys[1].Draw("e2 same")

                    if Ratio != None:
                        line.Draw("same")
                    Pad1.cd()

                    continue
            
            if inplots[i] ==None:
                continue
            inplots[i].SetMarkerColor(self.colorlist[ncolor])
            #inplots[i].SetFillColor(self.colorlist[ncolor])
            inplots[i].SetLineColor(self.colorlist[ncolor])
            inplots[i].GetXaxis().SetTitle(axisname[0])
            inplots[i].GetYaxis().SetTitle(axisname[1])

            if (self.doRescale and not(self.doLogY)):
                inplots[i].GetYaxis().SetRangeUser(0, maxval*4.5/3.)
            if (self.doRescale and self.doLogY):
                inplots[i].GetYaxis().SetRangeUser(minval/100., maxval*10.)
            ncolor+=1            

        count = 0

        for i in range(len(inplots)):
            if inplots[i] ==None:
                continue
            inplots[i].SetTitle("")
            if count != 0:
                self.drawOption = "same"
            if inplots[i].ClassName() != "TH2D":
                inplots[i].Draw(self.drawOption)
            else:
                inplots[i].Draw("colz") 

            count +=1

        count = 0

        for i in range(len(instacks)):

            if instacks[i] == None:
                continue

            if "TH2" not in (instacks[i].ClassName()):
                #instacks[i].GetYaxis().SetMaxDigits(3)
                if count == 0:
                    instacks[i].SetTitle("")
                    if self.CompareData:
                        instacks[i].GetXaxis().SetLabelSize(0)
                        instacks[i].GetXaxis().SetTitle("")
                    instacks[i].Draw("hist")
                    count +=1
                    continue
                else:
                    if self.CompareData:
                        XaxisTitle = instacks[i].GetXaxis().GetTitle()
                        labelsize = instacks[i].GetXaxis().GetLabelSize()
                        instacks[i].SetTitle("")
                        instacks[i].GetXaxis().SetLabelSize(0)

                        if  i == len(instacks)-1:
                            instacks[i].Draw("e same")
                            instacks[i].SetLineColor(1)
                            instacks[i].SetLineWidth(2)
                            instacks[i].SetMarkerColor(1)
                            instacks[i].SetFillColor(0)

                            #### pay attention ##
                            DataIntError = ROOT.Double(0)
                            DataInt = instacks[i].IntegralAndError(0, instacks[i].GetNbinsX()+1, DataIntError)
                            MCIntError  = ROOT.Double(0)
                            MCInt = instacks[i-1].IntegralAndError(0, instacks[i-1].GetNbinsX()+1, MCIntError)
                            if MCInt ==0:
                                MCInt = 1
                                print (" warning: no mc")
                            if DataInt ==0:
                                DataInt = 1
                                print (" warning: no data")
                            rat = DataInt/MCInt
                            rerr = sqrt((DataIntError/DataInt)**2+(MCIntError/MCInt)**2)*rat
                            #Atlas.myText(0.7, 0.45, 1, 0.04, "Data / MC = " + str(round(rat, 3)) +"#pm" + str(round(rerr,3)))
                            count +=1
                            #continue

                        if  i == len(instacks)-2:
                            instacks[i].SetMarkerStyle(10)
                            instacks[i].SetFillStyle(3004)
                            instacks[i].SetMarkerSize(0.00001)
                            instacks[i].SetFillColor(1)
                            instacks[i].SetLineWidth(0)
                            instacks[i].Draw("e2 same")
                            #print instacklabel[i], len(instacklabel)

                            Pad2.cd()
                            relsize = Pad2.GetAbsHNDC()/Pad1.GetAbsHNDC()
                            size = Atlas.tsize/relsize

                            Ratio = CopyHist (instacks[len(instacks)-1])
                            line = 0
                            if Ratio == None:
                                continue
                            if Ratio != None:
                                line = ROOT.TLine(Ratio.GetXaxis().GetBinLowEdge(1), 1, Ratio.GetXaxis().GetBinUpEdge(Ratio.GetNbinsX()), 1)
                            Ratio.Divide(instacks[i])

                            Ratio.SetTitle("")
                            Ratio.GetXaxis().SetLabelSize(size)
                            Ratio.GetYaxis().SetLabelSize(size)
                            Ratio.GetXaxis().SetTitleSize(size)
                            Ratio.GetYaxis().SetTitleSize(size) 
                            Ratio.GetXaxis().SetTitleOffset( Ratio.GetXaxis().GetTitleOffset()*relsize*2.9)
                            Ratio.GetXaxis().SetLabelOffset(0.03)
                            Ratio.GetYaxis().SetTitleOffset( Ratio.GetYaxis().GetTitleOffset()*relsize)
                            Ratio.GetYaxis().SetTitle(self.DrawRatio)
                            Ratio.GetXaxis().SetTitle(XaxisTitle)
                            Ratio.GetYaxis().SetRangeUser(0.5, 1.5)
                            Ratio.GetYaxis().SetNdivisions(4)
                            Ratio.SetMarkerColor(1)
                            Ratio.SetLineColor(1)
                            Ratio.SetLineWidth(2)
                            Ratio.GetYaxis().SetNdivisions(5, ROOT.kFALSE)
                            if sys ==[]:
                                Ratio.Draw('e')

                            elif len(sys)>1:

                                Ratio.Draw('e')

                                sys[1].SetFillStyle(3001)
                                sys[1].SetFillColor(30)
                                sys[1].SetMarkerStyle(10)
                                sys[1].SetMarkerSize(0)
                                sys[1].SetLineColor(2)
                                sys[1].SetLineWidth(2)
                                sys[1].Draw("e2 same")

                                sys[0].SetFillStyle(3004)
                                sys[0].SetFillColor(1)
                                sys[0].SetMarkerStyle(10)
                                sys[0].SetLineWidth(2)
                                sys[0].SetMarkerSize(0)
                                if sys[0].GetBinError(1)!= 0:
                                    sys[0].Draw("e2 same")
                                Ratio.Draw('e same')

                            else:
                                Ratio.Draw('e')
                                sys[0].SetFillStyle(3004)
                                sys[0].SetFillColor(1)
                                sys[0].SetMarkerStyle(10)
                                sys[0].SetLineWidth(2)
                                sys[0].SetMarkerSize(0)
                                sys[0].Draw("e2 same")


#                            if Ratio != None:
#                                line.Draw("same")
                            Pad1.cd()
                            count +=1
                            continue
                        
                        instacks[i].Draw("hist same")


                    else:
                        if  i == len(instacks)-1:
                            instacks[i].SetMarkerStyle(10)
                            instacks[i].SetFillStyle(3020)
                            instacks[i].Draw("e2 same")
                            count +=1
                            continue
                        instacks[i].Draw("hist same")

            ##### draw 2d
            else:
                instacks[i].SetTitle("")
                instacks[i].Draw('cont') 
                count +=1

        if( (instacks != [] and instacks[-1]!=None and ("TH2" not in (instacks[-1].ClassName())) ) or (inplots !=[] and inplots[-1]!=None and ("TH2" not in (inplots[-1].ClassName()))) ):
            legend.Draw("same")

        for text in self.texts:
            Atlas.myText(text[0], text[1] ,text[2], text[3], text[4])

        if self.doAtlasLabel:
            Atlas.ATLASLabel(self.AtlasLabelPos, 0.88, self.shift, self.studytype,color=1)
        if self.doLabel and self.lumi != "0":
            Atlas.myText(self.AtlasLabelPos, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV " + "#intLdt=" + self.lumi + " fb^{-1}") 
        if self.doLabel and self.lumi == "0":
            Atlas.myText(self.AtlasLabelPos, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV") 

        if self.doLogY:
            Canv.SetLogy()
        if self.doLogX:
            Canv.SetLogx()
        if self.doLogZ:
            Canv.SetLogz()

        Canv.Write()
        if (self.doPrintPlots):
            Canv.SaveAs(self.OutPlotDir+Canv.GetName()+".png")
        Canv.Close()

        return inplots


    def DrawHistsReweight(self, title, axisname=[], inplots =[], inlabel=[], instacks=[], instacklabel = [], sys = []):
        newinplots = []
        newinstacks = []

        for i in range(len(inplots)):
            newinplots.append(CopyHist(inplots[i]))
        for i in range(len(instacks)):
            newinstacks.append(CopyHist(instacks[i]))

        self.SetDrawScale(1.0)
        self.DrawHists(title, axisname, inplots, inlabel, instacks, instacklabel, sys)
        self.ClearTexts()

        ## reweight MC to data
        self.SetDrawScale(self.MCscale)
        self.DrawHists(title+"_MCScale", axisname, newinplots, inlabel, newinstacks, instacklabel, sys)
        self.SetDrawScale(1.0)
        self.ClearTexts()


    def DrawHistsWithSys(self, title, axisname=[], inplots =[], inlabel=[], sys = [], syslabel=[]):
        maxval =1
        minval =1000
        secminval = 10000
        legend = ROOT.TLegend(self.legend[0], self.legend[1], self.legend[2], self.legend[3])
        legend.SetFillColor(0)

        ## finding the right axises space         
        for i in range(len(inplots)):
            if inplots[i] == None:
                continue
            doExist = False

            legend.AddEntry(inplots[i], inlabel[i], "LPS")
            thismax = inplots[i].GetMaximum()
            thismin = inplots[i].GetMinimum()

            if maxval< thismax:
                maxval = thismax
            if (minval >= thismin):
                minval = thismin

        if minval <= 1.0:
            minval = 1.0

        ###### draw histogram
        Canv = ROOT.TCanvas('Canv_' + title, 'Canv_' + title, 0, 0, 800, 600)
        if(self.CompareData):
            Pad1 = ROOT.TPad('Pad1', 'Pad1', 0.0, 0.25, 1.0, 0.99, 0)
            Pad2 = ROOT.TPad('Pad2', 'Pad2', 0.0, 0.00, 1.0, 0.32, 0)
            Pad2.SetBottomMargin(0.4)
            Pad1.Draw()
            Pad2.Draw();
            Pad1.cd()


        ncolor =0
        count = 0

        for i in range(len(inplots)):
            if inplots[i] ==None:
                continue

            inplots[i].SetMarkerColor(self.colorlist[ncolor])
            inplots[i].SetFillColor(self.colorlist[ncolor])
            inplots[i].SetLineColor(self.colorlist[ncolor])
            inplots[i].GetXaxis().SetTitle(axisname[0])
            inplots[i].GetYaxis().SetTitle(axisname[1])

            if (self.doRescale and not(self.doLogY)):
                inplots[i].GetYaxis().SetRangeUser(0, maxval*4.5/3.)
            if (self.doRescale and self.doLogY):
                inplots[i].GetYaxis().SetRangeUser(minval/100., maxval*10.)
            ncolor+=1            

            inplots[i].SetTitle("")
            if count != 0:
                self.drawOption = "same"
                inplots[i].Draw("e"+self.drawOption)
            else:
                inplots[i].Draw("e")

            count +=1

        for i in range(len(sys)):
            sys[i].SetFillStyle(3004)
            sys[i].SetFillColor(self.colorlist[ncolor])            
            sys[i].SetLineColor(0)
            sys[i].SetMarkerStyle(10)
            sys[i].SetMarkerSize(0)
            sys[i].Draw("e2 same")
            ncolor+=1            
            count +=1
            legend.AddEntry(sys[i], syslabel[i], 'f')

        legend.Draw("same")

        for text in self.texts:
            Atlas.myText(text[0], text[1] ,text[2], text[3], text[4])

        if self.doAtlasLabel:
            Atlas.ATLASLabel(self.AtlasLabelPos, 0.88, 0.13, self.studytype,color=1)
        #if self.doLabel and self.lumi != "0":
        Atlas.myText(self.AtlasLabelPos, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV: " + "#intLdt=20.3" + " fb^{-1}") 
        #if self.doLabel and self.lumi == "0":
        #    Atlas.myText(0.2, 0.81 ,color=1, size=0.04,text="#sqrt{s}="+self.sqrtS + " TeV") 

        if self.doLogY:
            Canv.SetLogy()
        if self.doLogX:
            Canv.SetLogx()
        if self.doLogZ:
            Canv.SetLogz()

        Canv.Write()
        if (self.doPrintPlots):
            Canv.SaveAs(self.OutPlotDir+Canv.GetName()+".png")
        Canv.Close()

        return inplots
