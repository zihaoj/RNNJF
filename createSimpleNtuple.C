#include "readNtuple.h"
#include "TTree.h"
#include "TChain.h"
#include "TFile.h"
#include <vector>
#include "TLorentzVector.h"
#include <iostream>

using namespace std;


void createSimpleNtuple()
{
  

//create new info


//read info

  TChain* chain=new TChain("bTag_AntiKt4EMTopoJets");

  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000001.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000002.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000003.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000004.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000005.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000006.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000007.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000008.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000009.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000010.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000011.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000012.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000013.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000014.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000015.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000016.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000017.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000018.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000019.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000020.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000021.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000022.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000023.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000024.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000025.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000026.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000027.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000028.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000029.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000030.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000031.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000032.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000033.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000034.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000035.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000036.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000037.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000038.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000039.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000040.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000041.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000042.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000043.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000044.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000045.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000046.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000047.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000048.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000049.root");
  chain->AddFile("/eos/atlas/user/g/giacinto/MLWorkshopMay24/user.giacinto.11401650.Akt4EMTo._000050.root");

  assert(chain!=0);
  
  readNtuple myReader(chain);

//prepare output


  TFile* outputFile=new TFile("MLvertexTree.root","recreate");
  TTree* newTree=new TTree("MLTree","MLTree");
  
  //Sz , Sd , pfrac, ∆R(track, jet)

  vector<float> Sz;
  vector<float> Sd;
  vector<float> pTfrac;
  vector<float> DRtojet;
  vector<int> IP3Dcat;
  //new variables
  vector<float> dist3d;
  vector<float> dist2d;
  vector<float> err3d;
  vector<float> signif3d;
  vector<float> JFcompatibility;
//  vector<float> DRtoJFaxis;
  vector<float> DPhitoJFaxis;
  vector<float> DThetatoJFaxis;
  vector<float> momPerpJFaxis;
  int truthLabel;
  float jetPt;
  float jetEta;
  float weightIP3D;
  
  newTree->Branch("Sz",&Sz);
  newTree->Branch("Sd",&Sd);
  newTree->Branch("pTfrac",&pTfrac);
  newTree->Branch("DRtojet",&DRtojet);
  newTree->Branch("IP3Dcat",&IP3Dcat);
//new variables
  newTree->Branch("dist3d",&dist3d);
  newTree->Branch("dist2d",&dist2d);
  newTree->Branch("err3d",&err3d);
  newTree->Branch("signif3d",&signif3d);
  newTree->Branch("JFcompatibility",&JFcompatibility);
  newTree->Branch("DPhitoJFaxis",&DPhitoJFaxis);
  newTree->Branch("DThetatoJFaxis",&DThetatoJFaxis);
  newTree->Branch("momPerpJFaxis",&momPerpJFaxis);
  newTree->Branch("truthLabel",&truthLabel,"truthLabel/I");

  newTree->Branch("jetPt",&jetPt,"jetPt/F");
  newTree->Branch("jetEta",&jetEta,"jetEta/F");
  newTree->Branch("weightIP3D",&weightIP3D,"weightIP3D/F");




//now port to read to output

  int nEntries=chain->GetEntries();

  cout << " Total number of entries: " << nEntries << endl;
  
  
//  nEntries=10000;
  

  for (int i=0;i<nEntries;i++)
  {

    if (i % 10000 ==0) cout << " Processing entry " << i << endl;

    myReader.GetEntry(i);

    int nJets=myReader.njets;

    for (int iJet=0;iJet<nJets;iJet++)
    {

      if (myReader.jet_pt->at(iJet)<20000.) continue;
      if (fabs(myReader.jet_eta->at(iJet))>2.5) continue;
      if (fabs(myReader.jet_eta->at(iJet))<2.4 && myReader.jet_pt->at(iJet)<60e3 && 
          myReader.jet_JVT->at(iJet)<=0.59) continue;
      if (myReader.jet_aliveAfterOR->at(iJet)!=1) continue;

      truthLabel=myReader.jet_LabDr_HadF->at(iJet);

      jetPt=myReader.jet_pt->at(iJet);
      jetEta=myReader.jet_eta->at(iJet);
      weightIP3D=myReader.jet_ip3d_llr->at(iJet);

      Sz.clear();
      Sd.clear();
      pTfrac.clear();
      DRtojet.clear();
      IP3Dcat.clear();

      dist3d.clear();
      dist2d.clear();
      err3d.clear();
      signif3d.clear();
      JFcompatibility.clear();
//      DRtoJFaxis.clear();
      DPhitoJFaxis.clear();
      DThetatoJFaxis.clear();
      momPerpJFaxis.clear();
            
      int nTracks=myReader.jet_trk_pt->at(iJet).size();

      for (int kTrack=0;kTrack<nTracks;kTrack++)
      {

        //for now only use tracks selected by IP3D
        if (myReader.jet_trk_ip3d_grade->at(iJet).at(kTrack)<=0) continue;
        
        Sz.push_back(myReader.jet_trk_ip3d_d0sig->at(iJet).at(kTrack));
        Sd.push_back(myReader.jet_trk_ip3d_z0sig->at(iJet).at(kTrack));
        pTfrac.push_back(myReader.jet_trk_pt->at(iJet).at(kTrack)/myReader.jet_pt->at(iJet));
        DRtojet.push_back(myReader.jet_trk_dr->at(iJet).at(kTrack));
        IP3Dcat.push_back(myReader.jet_trk_ip3d_grade->at(iJet).at(kTrack));
        
        //I want to store things per track!

        int iVertex=myReader.jet_trk_jf_Vertex->at(iJet).at(kTrack);

        TVector3 jetDirection;
        jetDirection.SetPtEtaPhi(myReader.jet_pt->at(iJet),
                                 myReader.jet_eta->at(iJet),
                                 myReader.jet_phi->at(iJet));

        TVector3 jetFitterDirection;
        jetFitterDirection.SetPtThetaPhi(1,myReader.jet_jf_theta->at(iJet),myReader.jet_jf_phi->at(iJet));

        if (iVertex!=-1)
        {
          double distance3d=myReader.jet_jf_vtx_L3D->at(iJet).at(iVertex);
          double distance2d=distance3d*sin(myReader.jet_jf_theta->at(iJet));
          double error3d=myReader.jet_jf_vtx_sig3D->at(iJet).at(iVertex);
          double significance3d=distance3d/error3d;
          double jetFitterCompatibility=TMath::Prob(myReader.jet_jf_vtx_chi2->at(iJet).at(iVertex),
                                                    myReader.jet_jf_vtx_ndf->at(iJet).at(iVertex));
          
          TVector3 trackMomDirection;
          trackMomDirection.SetPtThetaPhi(myReader.jet_trk_pt->at(iJet).at(kTrack),
                                          myReader.jet_trk_theta->at(iJet).at(kTrack),
                                          myReader.jet_trk_phi->at(iJet).at(kTrack));


//          double deltaR=jetFitterDirection.DeltaR(trackMomDirection);
          double deltaPhi=trackMomDirection.DeltaPhi(jetFitterDirection);
          double deltaTheta=trackMomDirection.Theta()-jetFitterDirection.Theta();
                    
          TVector3 trackMomentum;
          trackMomentum.SetPtThetaPhi(myReader.jet_trk_pt->at(iJet).at(kTrack),
                                      myReader.jet_trk_theta->at(iJet).at(kTrack),
                                      myReader.jet_trk_phi->at(iJet).at(kTrack));
          
          double momT=trackMomentum.Perp(jetFitterDirection);

          dist3d.push_back(distance3d);
          dist2d.push_back(distance2d);
          err3d.push_back(error3d);
          signif3d.push_back(significance3d);
          JFcompatibility.push_back(jetFitterCompatibility);
//          DRtoJFaxis.push_back(deltaR);
          momPerpJFaxis.push_back(momT);
          DPhitoJFaxis.push_back(deltaPhi);
          DThetatoJFaxis.push_back(deltaTheta);
        }//if (iVertex!=-1) 
        else 
        {

          dist3d.push_back(0);
          dist2d.push_back(0);
          err3d.push_back(0);
          signif3d.push_back(0);
          JFcompatibility.push_back(0);
//          DRtoJFaxis.push_back(0);
          momPerpJFaxis.push_back(0);
          DPhitoJFaxis.push_back(0);
          DThetatoJFaxis.push_back(0);
        }

      }//end of track iterations

      newTree->Fill();

    }//end iteration of n jets
  }//end iteration of n entries

  newTree->Write();
  outputFile->Write();
  
  
  
}


    
    

  


