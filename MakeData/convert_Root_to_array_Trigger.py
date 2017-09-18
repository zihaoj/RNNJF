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

Njets = 10000
Nevents = int(1.0*Njets / 4.0)  #assume 4 jets per event

outfileName = "test.pkl"

##################
print "- Making File List"

ROOTfileNames = glob.glob("/atlas/local/BtagOptimizationNtuples/BTrigger/flavntuple_20170809trkInfo_ttbar_step1.root")

firstFile = True
EventSum = 0

out_trk_arr = None
out_label_arr = None

for fname in ROOTfileNames:

    Nleft = 0
    if out_label_arr!= None and len(out_label_arr) > Njets:
        break

    Nleft = Nevents - EventSum
    print "Nleft", Nleft
    if Nleft <= 0:
        break        
        
    with root_open(fname) as f:
        print "- opened file = ", fname
  
        tree = f.bTag_TriggerJets
        EventSum = EventSum + (Nleft if Nleft < f.bTag_TriggerJets.GetEntries() else f.bTag_TriggerJets.GetEntries())

        ####################################################################################
        #per track info
        # comes out in crazy form.... lots of processing needed...
        # array of events, event = array of jets,  jet  = array of variable
        ####################################################################################
        print "- extracting per track info"
        sd0_raw =   rnp.tree2array(tree, "jet_trk_d0sig",  stop=Nleft).flatten()
        sz0_raw =   rnp.tree2array(tree, "jet_trk_z0sig",  stop=Nleft).flatten()
        #grade_raw = rnp.tree2array(tree, "jet_trk_ip3d_grade",  stop=Nleft).flatten()
        pt_raw =    rnp.tree2array(tree, "jet_trk_pt",  stop=Nleft).flatten()
        eta_raw =   rnp.tree2array(tree, "jet_trk_eta",  stop=Nleft).flatten()
        phi_raw =   rnp.tree2array(tree, "jet_trk_phi",  stop=Nleft).flatten()

        print "- sorting per track info"
        sd0_arr = np.array([], dtype=object).reshape((1,0))
        sz0_arr = np.array([], dtype=object).reshape((1,0))
        pt_arr = np.array([], dtype=object).reshape((1,0))
        pTFrac_arr = np.array([], dtype=object).reshape((1,0))
        eta_arr = np.array([], dtype=object).reshape((1,0))
        phi_arr = np.array([], dtype=object).reshape((1,0))
        deta_arr = np.array([], dtype=object).reshape((1,0))
        dphi_arr = np.array([], dtype=object).reshape((1,0))
        dR_arr = np.array([], dtype=object).reshape((1,0))
        #grade_arr = None

        for ievt in range(len(sd0_raw)):
            for ijet in range(len(sd0_raw[ievt])):
                ################ sorting variable #######################
                index_list = sorting_funcs.get_sort_index_list( sd0_raw[ievt][ijet].flatten(),  sort_type="absrev" )
                ################ sorting variable #######################

                sd0_sort =   np.array(sorting_funcs.sort_arrays_in_list(sd0_raw[ievt][ijet].flatten(), index_list))
                sz0_sort =   np.array(sorting_funcs.sort_arrays_in_list(sz0_raw[ievt][ijet].flatten(), index_list))
                pt_sort =    np.array(sorting_funcs.sort_arrays_in_list(pt_raw[ievt][ijet].flatten(), index_list))
                eta_sort =   np.array(sorting_funcs.sort_arrays_in_list(eta_raw[ievt][ijet].flatten(), index_list))
                phi_sort =   np.array(sorting_funcs.sort_arrays_in_list(phi_raw[ievt][ijet].flatten(), index_list))
                #grade_sort = sorting_funcs.sort_arrays_in_list(grade_raw[i].flatten(), index_list, mask_list)

                sd0_arr =   np.append (sd0_arr, 0)
                sd0_arr[-1] = sd0_sort

                sz0_arr =   np.append (sz0_arr, 0)
                sz0_arr[-1] = sz0_sort

                #grade_arr = grade_sort
                pt_arr =    np.append (pt_arr, 0)
                pt_arr[-1]  = pt_sort

                eta_arr =   np.append (eta_arr, 0)
                eta_arr[-1] = eta_sort

                phi_arr =   np.append (phi_arr, 0)
                phi_arr[-1] = phi_sort

        ####################################################################################
        #lables array
        # comes out as array of arrays  (NOT 2D array, but array of arrays)
        # convert to tuple for easy stacking into single array
        ####################################################################################
        print "- extracting jet info"
        jet_flav =      np.hstack(tuple(rnp.tree2array(tree, "jet_truthflav",  stop=Nleft)))        
        jet_pt =        np.hstack(tuple(rnp.tree2array(tree, "jet_pt",  stop=Nleft)))        
        jet_eta =       np.hstack(tuple(rnp.tree2array(tree, "jet_eta", stop=Nleft)))        
        jet_phi =       np.hstack(tuple(rnp.tree2array(tree, "jet_phi", stop=Nleft)))        
        
        pTFrac_arr = deepcopy(pt_arr)
        dR_arr = deepcopy(pt_arr)
        deta_arr = deepcopy(pt_arr)
        dphi_arr = deepcopy(pt_arr)

        print "jet shape", jet_pt.shape
        print "ptfrac shape", pTFrac_arr.shape
        if jet_pt.shape[0] != pTFrac_arr.shape[0]:
            print "moving on"
            continue

        for iJet in range(jet_pt.shape[0]):
            for iTrk in range(pTFrac_arr[iJet].shape[0]):
                pTFrac_arr[iJet][iTrk] = pTFrac_arr[iJet][iTrk]/jet_pt[iJet]
                dR_arr [iJet][iTrk] = sqrt( (eta_arr[iJet][iTrk]-jet_eta[iJet])**2+(phi_arr[iJet][iTrk]-jet_phi[iJet])**2 )

        label_arr = np.dstack( (jet_flav, jet_pt, jet_eta, jet_phi) )[0]
        trk_arr = np.dstack( (sd0_arr, sz0_arr, pTFrac_arr, dR_arr) )[0] #to be added later grade_arr, llr_arr, pt_arr, nInnHits_arr, nNextToInnHits_arr,

        if firstFile:
            out_trk_arr = trk_arr
            out_label_arr = label_arr
            firstFile= False

        else:
            out_trk_arr = np.vstack( (out_trk_arr, trk_arr) )
            out_label_arr = np.vstack( (out_label_arr, label_arr) ) 



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
