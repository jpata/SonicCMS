from FWCore.ParameterSet.VarParsing import VarParsing
# Forked from SMPJ Analysis Framework
import FWCore.ParameterSet.Config as cms
import os, sys, json

options = VarParsing("analysis")
options.register("address", "127.0.0.1", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("port", 8001, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("timeout", 30, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("params", "", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("threads", 1, VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("streams", 0,    VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("batchsize", 10,    VarParsing.multiplicity.singleton, VarParsing.varType.int)
options.register("modelname", "mlpf", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.register("mode", "Sync", VarParsing.multiplicity.singleton, VarParsing.varType.string)
options.parseArguments()

if len(options.params)>0:
    with open(options.params,'r') as pfile:
        pdict = json.load(pfile)
    options.address = pdict["address"]
    options.port = int(pdict["port"])
    print("server = "+options.address+":"+str(options.port))

# check mode
allowed_modes = {
    "Async": "MLPFProducerAsync",
    "Sync": "MLPFProducerSync",
    "PseudoAsync": "MLPFProducerPseudoAsync",
}
if options.mode not in allowed_modes:
    raise ValueError("Unknown mode: "+options.mode)

process = cms.Process('mlpftest')

#--------------------------------------------------------------------------------
# Import of standard configurations
#================================================================================
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/GeometryDB_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('100X_upgrade2018_realistic_v10')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(['file:step3_inMINIAODSIM.root']*100),
	duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

if len(options.inputFiles)>0: process.source.fileNames = options.inputFiles

################### EDProducer ##############################
process.mlpfProducer = cms.EDProducer(allowed_modes[options.mode],
    Client = cms.PSet(
        ninput  = cms.uint32(1000*15),
        noutput = cms.uint32(1000*12),
        batchSize = cms.uint32(options.batchsize),
        address = cms.string(options.address),
        port = cms.uint32(options.port),
        timeout = cms.uint32(options.timeout),
        modelName = cms.string(options.modelname),
    )
)

# Let it run
process.p = cms.Path(
    process.mlpfProducer
)

process.MessageLogger.cerr.FwkReport.reportEvery = 500
keep_msgs = ['MLPFProducer','TRTClient']
for msg in keep_msgs:
    process.MessageLogger.categories.append(msg)
    setattr(process.MessageLogger.cerr,msg,
        cms.untracked.PSet(
            optionalPSet = cms.untracked.bool(True),
            limit = cms.untracked.int32(10000000),
        )
    )

if options.threads>0:
    if not hasattr(process,"options"):
        process.options = cms.untracked.PSet()
    process.options.numberOfThreads = cms.untracked.uint32(options.threads)
    process.options.numberOfStreams = cms.untracked.uint32(options.streams if options.streams>0 else 0)

process.Timing = cms.Service("Timing",
   summaryOnly = cms.untracked.bool(True),
   useJobReport = cms.untracked.bool(True)
 )
