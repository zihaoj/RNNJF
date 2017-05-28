# this import is required
import rootpy.tree
from rootpy.io import root_open
from ROOT import gDirectory, TLorentzVector, TVector3
  
#from rootpy.root2array import tree_to_ndarray
import root_numpy as rnp
import sys
import cPickle 
import numpy as np
import glob
import sorting_funcs 
from copy import deepcopy
from math import sqrt
import h5py

Njets = 1000000
Nevents = Njets

outfileName = "Dataset_JF_Giacinto_large.pkl"
#outfileName = "Dataset_JF_Giacinto.h5"

##################
print "- Making File List"

ROOTfileNames = glob.glob("/atlas/local/BtagOptimizationNtuples/GiacintoPrivate_17_05_25/*.root")

firstFile = True
EventSum = 0

out_trk_arr = None
out_label_arr = None

for fname in ROOTfileNames:

    Nleft = 0
    if out_label_arr!= None and len(out_label_arr) > Njets:
        break

    Nleft = Nevents - EventSum
    if Nleft <= 0:
        break        
        
    with root_open(fname) as f:
        print "- opened file = ", fname
  
        tree = f.MLTree

        EventSum = EventSum + (Nleft if Nleft < f.MLTree.GetEntries() else f.MLTree.GetEntries())

        ####################################################################################
        #per track info
        # comes out in crazy form.... lots of processing needed...
        # array of events, event = array of jets,  jet  = array of variable
        ####################################################################################
        print "- extracting per track info"
        Sd_raw =   rnp.tree2array(tree, "Sd",  stop=Nleft).flatten()
        Sz_raw =   rnp.tree2array(tree, "Sz",  stop=Nleft).flatten()
        pTfrac_raw  =   rnp.tree2array(tree, "pTfrac",  stop=Nleft).flatten()
        DRtojet_raw =   rnp.tree2array(tree, "DRtojet",  stop=Nleft).flatten()
        IP3Dcat_raw =   rnp.tree2array(tree, "IP3Dcat",  stop=Nleft).flatten()
        dist3d_raw =   rnp.tree2array(tree, "dist3d",  stop=Nleft).flatten()
        dist2d_raw =   rnp.tree2array(tree, "dist2d",  stop=Nleft).flatten()
        err3d_raw  =   rnp.tree2array(tree, "err3d",  stop=Nleft).flatten()
        signif3d_raw  =   rnp.tree2array(tree, "signif3d",  stop=Nleft).flatten()
        JFcompatibility_raw  =   rnp.tree2array(tree, "JFcompatibility",  stop=Nleft).flatten()
        DPhitoJFaxis_raw  =   rnp.tree2array(tree, "DPhitoJFaxis",  stop=Nleft).flatten()
        DThetatoJFaxis_raw  =   rnp.tree2array(tree, "DThetatoJFaxis",  stop=Nleft).flatten()
        momPerpJFaxis_raw  =   rnp.tree2array(tree, "momPerpJFaxis",  stop=Nleft).flatten()

        print "- sorting per track info"
        Sd_arr = np.array([], dtype=object).reshape((1,0))
        Sz_arr = np.array([], dtype=object).reshape((1,0))
        pTfrac_arr  =   np.array([], dtype=object).reshape((1,0))
        DRtojet_arr =   np.array([], dtype=object).reshape((1,0))
        IP3Dcat_arr =   np.array([], dtype=object).reshape((1,0))
        dist3d_arr =   np.array([], dtype=object).reshape((1,0))
        dist2d_arr =   np.array([], dtype=object).reshape((1,0))
        err3d_arr  =   np.array([], dtype=object).reshape((1,0))
        signif3d_arr  =   np.array([], dtype=object).reshape((1,0))
        JFcompatibility_arr  = np.array([], dtype=object).reshape((1,0))
        DPhitoJFaxis_arr  =   np.array([], dtype=object).reshape((1,0))
        DThetatoJFaxis_arr  = np.array([], dtype=object).reshape((1,0))
        momPerpJFaxis_arr  =  np.array([], dtype=object).reshape((1,0))

        #print Sd_raw
        #print IP3Dcat_raw
        #print "njets", len(Sd_raw)

        for i in range(len(Sd_raw)):
            if i%10000 ==0:
                print "jet ", i

            ################ sorting variable #######################
            index_list = sorting_funcs.get_sort_index_list( Sd_raw[i].flatten(),  sort_type="absrev" )
            ################ sorting variable #######################

            mask_list = sorting_funcs.get_neg_mask_list( sorting_funcs.sort_arrays_in_list(IP3Dcat_raw[i].flatten(), index_list) )

            Sd_sort =   sorting_funcs.sort_arrays_in_list(Sd_raw[i].flatten(), index_list, mask_list)
            Sz_sort =   sorting_funcs.sort_arrays_in_list(Sz_raw[i].flatten(), index_list, mask_list)
            pTfrac_sort =   sorting_funcs.sort_arrays_in_list(pTfrac_raw[i].flatten(), index_list, mask_list)
            DRtojet_sort =   sorting_funcs.sort_arrays_in_list(DRtojet_raw[i].flatten(), index_list, mask_list)
            IP3Dcat_sort =   sorting_funcs.sort_arrays_in_list(IP3Dcat_raw[i].flatten(), index_list, mask_list)
            dist3d_sort =   sorting_funcs.sort_arrays_in_list(dist3d_raw[i].flatten(), index_list, mask_list)
            dist2d_sort =   sorting_funcs.sort_arrays_in_list(dist2d_raw[i].flatten(), index_list, mask_list)
            err3d_sort =   sorting_funcs.sort_arrays_in_list(err3d_raw[i].flatten(), index_list, mask_list)
            signif3d_sort =   sorting_funcs.sort_arrays_in_list(signif3d_raw[i].flatten(), index_list, mask_list)
            JFcompatibility_sort =   sorting_funcs.sort_arrays_in_list(JFcompatibility_raw[i].flatten(), index_list, mask_list)
            DPhitoJFaxis_sort =   sorting_funcs.sort_arrays_in_list(DPhitoJFaxis_raw[i].flatten(), index_list, mask_list)
            DThetatoJFaxis_sort =   sorting_funcs.sort_arrays_in_list(DThetatoJFaxis_raw[i].flatten(), index_list, mask_list)
            momPerpJFaxis_sort =   sorting_funcs.sort_arrays_in_list(momPerpJFaxis_raw[i].flatten(), index_list, mask_list)

            #print "Sd_sort", Sd_sort.shape



#            if i==0:
#                Sd_arr =   Sd_sort
#                Sz_arr =   Sz_sort
#                pTfrac_arr =   pTfrac_sort
#                DRtojet_arr =   DRtojet_sort
#                IP3Dcat_arr =   IP3Dcat_sort
#                dist3d_arr =   dist3d_sort
#                dist2d_arr =   dist2d_sort
#                err3d_arr =   err3d_sort
#                signif3d_arr =   signif3d_sort
#                JFcompatibility_arr =   JFcompatibility_sort
#                DPhitoJFaxis_arr =   DPhitoJFaxis_sort
#                DThetatoJFaxis_arr = DThetatoJFaxis_sort
#                momPerpJFaxis_arr =  momPerpJFaxis_sort
#
#            else:
            
            Sd_arr = np.append (Sd_arr, 0)
            Sd_arr[-1] = Sd_sort

            Sz_arr = np.append (Sz_arr, 0)
            Sz_arr[-1] = Sz_sort

            pTfrac_arr = np.append (pTfrac_arr, 0)
            pTfrac_arr[-1] = pTfrac_sort

            DRtojet_arr = np.append (DRtojet_arr, 0)
            DRtojet_arr[-1] = DRtojet_sort

            IP3Dcat_arr = np.append (IP3Dcat_arr, 0)
            IP3Dcat_arr[-1] = IP3Dcat_sort

            dist3d_arr = np.append (dist3d_arr, 0)
            dist3d_arr[-1] = dist3d_sort

            dist2d_arr = np.append (dist2d_arr, 0)
            dist2d_arr[-1] = dist2d_sort

            err3d_arr = np.append (err3d_arr, 0)
            err3d_arr[-1] = err3d_sort

            signif3d_arr = np.append (signif3d_arr, 0)
            signif3d_arr[-1] = signif3d_sort

            JFcompatibility_arr = np.append (JFcompatibility_arr, 0)
            JFcompatibility_arr[-1] = JFcompatibility_sort

            DPhitoJFaxis_arr = np.append (DPhitoJFaxis_arr, 0)
            DPhitoJFaxis_arr[-1] = DPhitoJFaxis_sort

            DThetatoJFaxis_arr = np.append (DThetatoJFaxis_arr, 0)
            DThetatoJFaxis_arr[-1] = DThetatoJFaxis_sort

            momPerpJFaxis_arr = np.append (momPerpJFaxis_arr, 0)
            momPerpJFaxis_arr[-1] = momPerpJFaxis_sort

        #print "Sd0_arr", Sd_arr.shape


        ####################################################################################
        #lables array
        # comes out as array of arrays  (NOT 2D array, but array of arrays)
        # convert to tuple for easy stacking into single array
        ####################################################################################
        print "- extracting jet info"

        jet_label =      np.hstack(tuple(rnp.tree2array(tree, "truthLabel",  stop=Nleft)))        
        jet_pt =        np.hstack(tuple(rnp.tree2array(tree, "jetPt",  stop=Nleft)))        
        jet_eta =       np.hstack(tuple(rnp.tree2array(tree, "jetEta", stop=Nleft)))        
        jet_IP3D =       np.hstack(tuple(rnp.tree2array(tree, "weightIP3D", stop=Nleft)))        

        label_arr = np.dstack( (jet_label, jet_pt, jet_eta, jet_IP3D) )[0]
        trk_arr = np.dstack( (  Sd_arr,
                                Sz_arr,
                                pTfrac_arr,
                                DRtojet_arr, 
                                IP3Dcat_arr,
                                dist3d_arr,
                                dist2d_arr,
                                err3d_arr,
                                signif3d_arr,
                                JFcompatibility_arr,
                                DPhitoJFaxis_arr,
                                DThetatoJFaxis_arr,
                                momPerpJFaxis_arr
                                ) )[0] 

        
        if firstFile:
            out_trk_arr = trk_arr
            out_label_arr = label_arr
            firstFile= False

        else:
            out_trk_arr = np.vstack( (out_trk_arr, trk_arr) )
            out_label_arr = np.vstack( (out_label_arr, label_arr) ) 
        
        print "out_trk_arr", out_trk_arr.shape
        print "out_label_arr", out_label_arr.shape


print "- finished, saving"

outfile = file(outfileName, 'wb')
cPickle.dump(out_trk_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
cPickle.dump(out_label_arr, outfile, protocol=cPickle.HIGHEST_PROTOCOL)
outfile.close()

print "######### Summary #########"
print "OutFile = ", outfileName
print "Nevents = ", EventSum
print "Njets = ", len(out_label_arr)
print "###########################"
