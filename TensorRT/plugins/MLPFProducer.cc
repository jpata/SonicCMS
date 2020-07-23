#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>

#include "SonicCMS/Core/interface/SonicEDProducer.h"
#include "SonicCMS/TensorRT/interface/TRTClient.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

#include "TLorentzVector.h"

//The model takes the following number of features for each input PFElement
static const unsigned int NUM_ELEMENT_FEATURES = 15;

//The model has 13 outputs for each particle, which consist of:

static const unsigned int NUM_OUTPUTS = 12;
//0...7: class probabilities
//8: eta
//9: phi
//10: energy
//11: charge
static const unsigned int NUM_CLASS = 7;
static const unsigned int IDX_ETA = 8;
static const unsigned int IDX_PHI = 9;
static const unsigned int IDX_ENERGY = 10;
static const unsigned int IDX_CHARGE = 11;

//index [0, N_pdgids) -> PDGID
//this maps the absolute values of the predicted PDGIDs to an array of ascending indices
static const std::vector<int> pdgid_encoding = {0, 1, 2, 11, 13, 22, 130, 211};

//PFElement::type -> index [0, N_types)
//this maps the type of the PFElement to an ascending index that is used by the model to distinguish between different elements
static const std::map<int, int> elem_type_encoding = {
    {0, 0},
    {1, 1},
    {2, 2},
    {3, 3},
    {4, 4},
    {5, 5},
    {6, 6},
    {7, 7},
    {8, 8},
    {9, 9},
    {10, 10},
    {11, 11},
};

template <typename T, typename A>
int arg_max(std::vector<T, A> const& vec) {
  return static_cast<int>(std::distance(vec.begin(), max_element(vec.begin(), vec.end())));
}

float normalize(float in) {
  if (std::abs(in) > 1e4) {
    return 0.0;
  } else if (std::isnan(in)) {
    return 0.0;
  }
  return in;
}

//Prepares the input array of floats for a single PFElement
std::array<float, NUM_ELEMENT_FEATURES> get_element_properties(const reco::PFBlockElement& orig) {
  const auto type = orig.type();
  float pt = 0.0;
  //these are placeholders for the the future
  [[maybe_unused]] float deltap = 0.0;
  [[maybe_unused]] float sigmadeltap = 0.0;
  [[maybe_unused]] float px = 0.0;
  [[maybe_unused]] float py = 0.0;
  [[maybe_unused]] float pz = 0.0;
  float eta = 0.0;
  float phi = 0.0;
  float energy = 0.0;
  float trajpoint = 0.0;
  float eta_ecal = 0.0;
  float phi_ecal = 0.0;
  float eta_hcal = 0.0;
  float phi_hcal = 0.0;
  float charge = 0;
  float layer = 0;
  float depth = 0;
  float muon_dt_hits = 0.0;
  float muon_csc_hits = 0.0;

  if (type == reco::PFBlockElement::TRACK) {
    const auto& matched_pftrack = orig.trackRefPF();
    if (matched_pftrack.isNonnull()) {
      const auto& atECAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::ECALShowerMax);
      const auto& atHCAL = matched_pftrack->extrapolatedPoint(reco::PFTrajectoryPoint::HCALEntrance);
      if (atHCAL.isValid()) {
        eta_hcal = atHCAL.positionREP().eta();
        phi_hcal = atHCAL.positionREP().phi();
      }
      if (atECAL.isValid()) {
        eta_ecal = atECAL.positionREP().eta();
        phi_ecal = atECAL.positionREP().phi();
      }
    }
    const auto& ref = ((const reco::PFBlockElementTrack*)&orig)->trackRef();
    pt = ref->pt();
    px = ref->px();
    py = ref->py();
    pz = ref->pz();
    eta = ref->eta();
    phi = ref->phi();
    energy = ref->pt() * cosh(ref->eta());
    charge = ref->charge();

    reco::MuonRef muonRef = orig.muonRef();
    if (muonRef.isNonnull()) {
      reco::TrackRef standAloneMu = muonRef->standAloneMuon();
      if (standAloneMu.isNonnull()) {
        muon_dt_hits = standAloneMu->hitPattern().numberOfValidMuonDTHits();
        muon_csc_hits = standAloneMu->hitPattern().numberOfValidMuonCSCHits();
      }
    }

  } else if (type == reco::PFBlockElement::BREM) {
    const auto* orig2 = (const reco::PFBlockElementBrem*)&orig;
    const auto& ref = orig2->GsftrackRef();
    if (ref.isNonnull()) {
      deltap = orig2->DeltaP();
      sigmadeltap = orig2->SigmaDeltaP();
      pt = ref->pt();
      px = ref->px();
      py = ref->py();
      pz = ref->pz();
      eta = ref->eta();
      phi = ref->phi();
      energy = ref->pt() * cosh(ref->eta());
      trajpoint = orig2->indTrajPoint();
      charge = ref->charge();
    }
  } else if (type == reco::PFBlockElement::GSF) {
    //requires to keep GsfPFRecTracks
    const auto* orig2 = (const reco::PFBlockElementGsfTrack*)&orig;
    pt = orig2->Pin().pt();
    px = orig2->Pin().px();
    py = orig2->Pin().py();
    pz = orig2->Pin().pz();
    eta = orig2->Pin().eta();
    phi = orig2->Pin().phi();
    energy = pt * cosh(eta);
    if (!orig2->GsftrackRefPF().isNull()) {
      charge = orig2->GsftrackRefPF()->charge();
    }
  } else if (type == reco::PFBlockElement::ECAL || type == reco::PFBlockElement::PS1 ||
             type == reco::PFBlockElement::PS2 || type == reco::PFBlockElement::HCAL ||
             type == reco::PFBlockElement::HO || type == reco::PFBlockElement::HFHAD ||
             type == reco::PFBlockElement::HFEM) {
    const auto& ref = ((const reco::PFBlockElementCluster*)&orig)->clusterRef();
    if (ref.isNonnull()) {
      eta = ref->eta();
      phi = ref->phi();
      px = ref->position().x();
      py = ref->position().y();
      pz = ref->position().z();
      energy = ref->energy();
      layer = ref->layer();
      depth = ref->depth();
    }
  } else if (type == reco::PFBlockElement::SC) {
    const auto& clref = ((const reco::PFBlockElementSuperCluster*)&orig)->superClusterRef();
    if (clref.isNonnull()) {
      eta = clref->eta();
      phi = clref->phi();
      px = clref->position().x();
      py = clref->position().y();
      pz = clref->position().z();
      energy = clref->energy();
    }
  }

  float typ_idx = static_cast<float>(elem_type_encoding.at(orig.type()));

  //Must be the same order as in tf_model.py
  return std::array<float, NUM_ELEMENT_FEATURES>({{typ_idx,
                                                   pt,
                                                   eta,
                                                   phi,
                                                   energy,
                                                   layer,
                                                   depth,
                                                   charge,
                                                   trajpoint,
                                                   eta_ecal,
                                                   phi_ecal,
                                                   eta_hcal,
                                                   phi_hcal,
                                                   muon_dt_hits,
                                                   muon_csc_hits}});
}
template <typename Client>
class MLPFProducer : public SonicEDProducer<Client> {
public:
  //needed because base class has dependent scope
  using typename SonicEDProducer<Client>::Input;
  using typename SonicEDProducer<Client>::Output;
  explicit MLPFProducer(edm::ParameterSet const& cfg)
      : SonicEDProducer<Client>(cfg),
        pfCandidatesToken_{this->template produces<reco::PFCandidateCollection>()},
        inputTagBlocks_(this->template consumes<reco::PFBlockCollection>(cfg.getParameter<edm::InputTag>("src"))) {
    //for debugging
    this->setDebugName("MLPFProducer");
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    edm::LogInfo("MLPFProducer") << "acquire";

    auto blocks_handle = iEvent.getHandle(inputTagBlocks_);
    const auto& blocks = *blocks_handle;

    iInput = Input(client_.ninput() * client_.batchSize(), 0.0f);
    edm::LogInfo("MLPFProducer") << "batchSize=" << client_.batchSize() << " ninput=" << client_.ninput();

    int ielem = 0;
    for (const auto& block : blocks) {
      const auto& elems = block.elements();
      for (const auto& elem : elems) {
        const auto& props = get_element_properties(elem);
        for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
          iInput[ielem * NUM_ELEMENT_FEATURES + iprop] = props.at(iprop);
        } //props
        ielem++;
      } //elems
    } //blocks
    edm::LogInfo("MLPFProducer") << "nblocks=" << blocks.size() << " nelem=" << ielem;

    //std::stringstream msg;
    for (unsigned int ibatch = 0; ibatch < client_.batchSize(); ibatch++) {
      for (unsigned int ielem = 0; ielem < 1000; ielem++) {
        //msg << "ibatch=" << ibatch << " ielem=" << ielem << " ";
        for (unsigned int iprop = 0; iprop < NUM_ELEMENT_FEATURES; iprop++) {
          //msg << iInput[ibatch*1000 + ielem*NUM_ELEMENT_FEATURES + iprop] << " ";
        }
        //msg << std::endl;
      }
    }
    //edm::LogInfo("MLPFProducer") << msg.str();

    edm::LogInfo("MLPFProducer") << "acquire done";
  }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    edm::LogInfo("MLPFProducer") << "produce";
    std::vector<reco::PFCandidate> pOutputCandidateCollection;

    edm::LogInfo("MLPFProducer") << "batchSize=" << client_.batchSize() << " noutput=" << client_.noutput();

    float output[client_.batchSize()][1000][NUM_OUTPUTS];
    
    for (unsigned int ibatch = 0; ibatch < client_.batchSize(); ibatch++) {
      for (unsigned int ielem = 0; ielem < 1000; ielem++) {
        for (unsigned int iprop = 0; iprop < NUM_OUTPUTS; iprop++) {
          output[ibatch][ielem][iprop] = iOutput.at(ibatch*1000 + ielem*NUM_OUTPUTS + iprop);
        } 
      }
    }
    std::stringstream msg;
    for (unsigned int ibatch=0; ibatch<client_.batchSize(); ibatch++) {
      for (unsigned int ielem=0; ielem<1000; ielem++) {
        std::vector<float> pred_id_logits;
        for (unsigned int idx_id=0; idx_id <= NUM_CLASS; idx_id++) {
          pred_id_logits.push_back(output[ibatch][ielem][idx_id]);
        }
        //get the most probable class PDGID
        int pred_pid = pdgid_encoding.at(arg_max(pred_id_logits));
        if (pred_pid != 0) {
          float pred_eta = output[ibatch][ielem][IDX_ETA];
          float pred_phi = output[ibatch][ielem][IDX_PHI];
          pred_phi = atan2(sin(pred_phi), cos(pred_phi));
          float pred_e = output[ibatch][ielem][IDX_ENERGY];
          float pred_charge = output[ibatch][ielem][IDX_CHARGE];
          float pred_pt = pred_e / cosh(pred_eta);
          reco::PFCandidate::Charge charge = 0;
          if (pred_pid == 11 || pred_pid == 13 || pred_pid == 211) {
              charge = pred_charge > 0 ? +1 : -1;
          }
          TLorentzVector p4;
          p4.SetPtEtaPhiE(pred_pt, pred_eta, pred_phi, pred_e);

          reco::PFCandidate cand(0, math::XYZTLorentzVector(p4.X(), p4.Y(), p4.Z(), p4.E()), reco::PFCandidate::ParticleType(0));
          cand.setPdgId(pred_pid);
          cand.setCharge(charge);
          pOutputCandidateCollection.push_back(cand);
        }
      }
    }
    iEvent.emplace(pfCandidatesToken_, pOutputCandidateCollection);

    edm::LogInfo("MLPFProducer") << msg.str();
    edm::LogInfo("MLPFProducer") << "produce done";
  }
  ~MLPFProducer() override {}

  //to ensure distinct cfi names - specialized below
  static std::string getCfiName();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    Client::fillPSetDescription(desc);
    desc.add<edm::InputTag>("src", edm::InputTag("particleFlowBlock"));
    descriptions.add(getCfiName(), desc);
  }

private:
  using SonicEDProducer<Client>::client_;

  const edm::EDPutTokenT<reco::PFCandidateCollection> pfCandidatesToken_;
  const edm::EDGetTokenT<reco::PFBlockCollection> inputTagBlocks_;
};

typedef MLPFProducer<TRTClientSync> MLPFProducerSync;
typedef MLPFProducer<TRTClientAsync> MLPFProducerAsync;
typedef MLPFProducer<TRTClientPseudoAsync> MLPFProducerPseudoAsync;

template <>
std::string MLPFProducerSync::getCfiName() {
  return "MLPFProducerSync";
}
template <>
std::string MLPFProducerAsync::getCfiName() {
  return "MLPFProducerAsync";
}
template <>
std::string MLPFProducerPseudoAsync::getCfiName() {
  return "MLPFProducerPseudoAsync";
}

DEFINE_FWK_MODULE(MLPFProducerSync);
DEFINE_FWK_MODULE(MLPFProducerAsync);
DEFINE_FWK_MODULE(MLPFProducerPseudoAsync);
