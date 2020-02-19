from pdb_dataset import PDBDataset
from pdbqt_dataset import PDBQTDataset
from molecule_contexts_dataset import MoleculeContextsDataset
from util import traverseDirectory

import os

class PreprocessDockingOutput:

    def __init__(self, 
                 dockingProgram = 'vina'
                 ):
        """
        Constructor.
        """
        self.dockingProgram = dockingProgram
        
    def loadActiveLigands(self, fileName):
        """
        Loads active ligands.
        """
        ligands = {}
        #print(fileName)
        f = open(fileName, "r")
        for line in f:
            #print(line)
            l = line.strip().split("_")[0]
            if len(l) > 1:
                lName = l
                k = 2
                while ligands.get(lName) != None:
                    lName = "%s%d"%(l,k)
                    k += 1
                ligands[lName] = True
        f.close()
        return ligands

    def processAllComplexesOfProtein(self, dataPath, proteinName, outputFileName,
                                     numIntraNeighbors, numInterNeighbors, 
                                     distIntervSize=.3, distMax=5, 
                                     chargeIntervSize=.05, chargeMin=-1, chargeMax=1):
        """
        Loads the contexts of all ligands and decoys of a given protein.
        """
        print("Loading ligands of", proteinName)
        #print(dataPath)
        activeLigandsFileName = os.path.join(dataPath, "%s/ligands.list"%proteinName)
        #print(activeLigandsFileName)
        activeLigands = self.loadActiveLigands(activeLigandsFileName)
        print(activeLigands)

        print("Loading protein and molecules of", proteinName)
        if self.dockingProgram == 'dock':
            proteinFileName       = os.path.join(dataPath, "%s/rec.mol2"%proteinName)
            moleculesFileName     = os.path.join(dataPath, "%s/virtual_flex.mol2"%proteinName)
            dsMolecules = PDBDataset()
            
        elif self.dockingProgram == 'vina':
            proteinFileName       = os.path.join(dataPath, "%s/rec.pdb"%proteinName)
            moleculesFileName     = os.path.join(dataPath, "%s/dock_out"%proteinName)
            dsMolecules = PDBQTDataset()
			
           
        dsMolecules.load(proteinFileName)   # the protein must be the first item in the loaded dataset
     #   print(proteinFileName)
        dsMolecules.load(moleculesFileName)
     #   print(moleculesFileName)
		
        print("Creating KDTrees...")
        dsMolecules.createMoleculeKDTrees()

        print("Creating contexts of protein:", proteinName)
        molContextsDs = MoleculeContextsDataset(
                                        distanceDiscretizationIntervalSize = distIntervSize,
                                        largestDistance = distMax,
                                        chargeDiscretizationIntervalSize = chargeIntervSize,
                                        largestCharge  = chargeMax,
                                        smallestCharge = chargeMin)
        
        molContextsDs.createExamples(molecules = dsMolecules, 
                                     proteinId = 0, # the protein is the first molecule in the loaded dataset
                                     moleculeIdStart= 1, 
                                     moleculeIdEnd  = dsMolecules.getNumberOfMolecules(),  
                                     numIntraNeighbors = numIntraNeighbors, 
                                     numInterNeighbors = numInterNeighbors,
                                     ligands = activeLigands)
        
        molContextsDs.discretizeDistance()
        molContextsDs.discretizeCharges()
        
        molContextsDs.saveToFile(outputFileName)


    def loadVinaScores(self, proteinPath):
        '''
        Loads Vina scores. 
        '''
        self.vinaScoreAndMoleculeName = []
        traverseDirectory(os.path.join(proteinPath, "vina_out"), callback=self.extractScoreFromPDBQTFile, 
                              extension=".pdbqt")
            

    def extractScoreFromPDBQTFile(self, fileName):
        """
        Loads the scores from VINA output.
        """
        f = open(fileName, "r")
        #removes _1.pdbqt
        moleculeName = os.path.basename(fileName)[:-8]
        score = 0
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                score = float(line[len("REMARK VINA RESULT:"):].strip().split()[0])
                break
        f.close()
        self.vinaScoreAndMoleculeName.append((score, moleculeName, len(self.vinaScoreAndMoleculeName)))
                
if __name__ == '__main__':
    dataPath  = '../dud_vinaout_deepvs/'
    # DeepVS-master/dud_vinaout_deepvs/
    numIntraNeighbors    = 6
    numInterNeighbors    = 2
    distanceIntervalSize = .3
    distanceMax          = 5 
    chargeIntervalSize   = .05
    chargeMin            = -1
    chargeMax            = 1
    
    proteinsToProcess = ['3lxl', '3max',
	'3mhw', '3mj1', '3mpm', '3npc', '3nu3',
	'3nw7', '3ny9', '3oll', '3pp0', '3qkl',
	'3r04', '3rm2', '3sff', '3skc', '3v8s']
	#['1a5h', '3tfq', '1a4g', '3klm', '1adw',
	#'1ah3', '1b8o', '1ckp', '1cx2', '1e3g',
	#'1eve', '1f0r', '1fm9', '1hw8', '1i00',
	#'1j8f', '1m17', '1nhz', '1ouk', '1p44',
	#'1r4l', '1s3v', '1s6p', '1uou', '1uy6',
	#'1uze', '1w4r', '1xjd', '1xoi', '1xp0',
	#'1z11', '2afx', '2b1p', '2dg3', '2iwi',
	#'2oo8', '2p1t', '2p54', '2src', '2vgo',
	#'2vwz', '2w31', '2w8y', '2wcg', '2xch',
	#'2z94', '3bc3', '3c7q', '3dbs', '3dds',
	#'3elj', '3eml', '3ewj', '3fdn', '3frg',
	#'3hng', '3i4b', '3k5e', '3kc3', '3kk6',
	#'3kx1', '3l3m', '3lbk', 
    proteinNames = []
    for p in proteinsToProcess:
        if(os.path.isdir(dataPath+p)):
            proteinNames.append(p)
						  # ['ace', 'ache', 'ada',
	
						  #	'alr2', 'ampc', 'ar', 'cdk2','comt', 'cox1', 
                          #  'cox2', 'dhfr', 'egfr', 'er_agonist', 'er_antagonist', 
                          #  'fgfr1', 'fxa', 'gart', 'gpb', 'gr', 'hivpr', 'hivrt', 'hmga', 'hsp90', 'inha', 'mr', 
                          #  'na', 'p38', 'parp', 'pde5', 'pdgfrb', 'pnp', 'ppar', 'pr', 'rxr', 'sahh', 
                          #  'src', 'thrombin', 'tk', 'trypsin', 'vegfr2']
    
    
    preprocessor = PreprocessDockingOutput('vina')
    for proteinName in proteinNames:
        outputFileName = os.path.join(dataPath, "%s.deepvs"%proteinName)
        preprocessor.processAllComplexesOfProtein(dataPath, 
                                         proteinName, 
                                         outputFileName,
                                         numIntraNeighbors, numInterNeighbors, 
                                         distIntervSize=distanceIntervalSize, 
                                         distMax=distanceMax, 
                                         chargeIntervSize=chargeIntervalSize, 
                                         chargeMin=chargeMin, 
                                         chargeMax=chargeMax)
