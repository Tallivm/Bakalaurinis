#! -*- coding: UTF-8 -*-
from pdb_dataset import PDBDataset
from util import traverseDirectory

import os
import numpy
from codecs import open


class PDBQTDataset(PDBDataset):
    """
    This class implements a dataset that can handle PDBQT files.
    """

    def load(self, datasetFileName):
        ''' 
        Loads the dataset
    
        :type datasetFileName: string
        :param datasetFileName: the path to the dataset
        '''
        if (not os.path.exists(datasetFileName)):
            raise Exception("%s is not a valid directory name."%datasetFileName)
        
        if(datasetFileName.endswith('.pdb')):
            traverseDirectory(datasetFileName, self.loadPDBQTFile, extension=".pdb")
        else: 
            #print(datasetFileName)
            traverseDirectory(datasetFileName, self.loadPDBQTFile, extension=".dok")
			
    def loadPDBQTFile(self, datasetFileName):
        ''' 
        Loads one pdb/dok file
    
        :type datasetFileName: string
        :param datasetFileName: the path to the dataset
        '''
        print("DATASET ------------> ", datasetFileName)
        if(str(datasetFileName).__contains__("ZINC")):
            ending = "_decoy"
            path = "decoys_mol2/"
            #moleculeName = os.path.basename(datasetFileName)[:-5]
        else:
            path = "ligands_seperate/"
            ending = "_active"
        fin  = open(datasetFileName,'r')

        line  = fin.readline()
        #print(line)
        atomsNames      = []
        atoms           = []
        atomPositions   = []
        atomAminoAcides = []
        atomCharges     = []
        #atomBonds       = []
        moleculeName = os.path.basename(datasetFileName).split("_")[0]
        #print(moleculeName)
        i = 0
        while line:
            line = line.strip()
            if line == "END":
                break
            # reads data of one atom
            if line.startswith("ATOM "):
                # 12 fields: [keyword, id, atom_name, amino, ?, x, y, z, ?, ?, charge, ?]
                data = line.split()
                #print(data)
                # reads cleaned atom name (without numbers)
                atoms.append(self.getTermIndexAdd(self.cleanAtomName(data[2])))
                # reads atom name
                atomsNames.append(self.getTermIndexAdd(data[2]))
                # reads atom position
                if(datasetFileName.endswith('.pdb')):
                      atomPositions.append([float(data[6]), float(data[7]), float(data[8])])
                      try:
                    # reads the charge
                            #print(data)
                            if(len(data)==11):
                                atomCharges.append(float(data[9]))
                            elif(len(data)==12):
                                atomCharges.append(float(data[10]))
                            else:
                                atomCharges.append(float(data[11]))
                            #print(len(data))
                      except ValueError:
                    # sometimes the field 8 and 9 are collapsed, 
                    # in these cases we only have 11 fields
                            print("Error in:", datasetFileName)
                            print("Line:", line)
                            #print(len(data))
                            #print(len(data))
                            if len(data) == 12:
                                atomCharges.append(float(data[10]))
                                #print("Added charge: ", data[10])
                            else:
                                atomCharges.append(0.0)
                                #print('Added 0.0 as charge: ')
                else: # .dok file xyz positions have different index
                    atomPositions.append([float(data[5]), float(data[6]), float(data[7])])
                    #print(datasetFileName)
                    #molNameExt = datasetFileName.split('deepvs/',1)[1]
                    #molName = molNameExt.split("/", 1)[0]	
                    #ligfolder = "/ligand_charges/"
                    #chargepath = "../dud_vinaout_deepvs/" + molName + ligfolder + molName + ending +".mol2"
                    molNameExt = datasetFileName.split('\\',1)[1]
                    molName = molNameExt.split(".")[0]
                    chargepath = "../dud_vinaout_deepvs/" + path + molName + ".mol2"
                    charge = PDBDataset().load(chargepath, moleculeName, i)
                    #print("Charge list: ",charge)
                    atomCharges.append(float(charge))
                    #print(atomCharges[i])
                    i = i + 1					
                    	
				
				
				
				# reads the amino acid name
                atomAminoAcides.append(self.getTermIndexAdd(data[3]))
				
				
    #           try:
    #                # reads the charge
    #                atomCharges.append(float(data[10]))
    #            except ValueError:
    #                # sometimes the field 8 and 9 are collapsed, 
    #                # in these cases we only have 11 fields
    #                print("Error in:", datasetFileName)
    #                print("Line:", line)
    #                if len(data) == 11:
    #                    atomCharges.append(float(data[9]))
    #                else:
    #                    atomCharges.append(0.0)
            
            line = fin.readline()
        fin.close()

        if len(atoms) > 0:
            self._moleculeAtoms.append(numpy.asarray(atoms,numpy.int32))
            self._moleculeAtomNames.append(numpy.asarray(atomsNames,numpy.int32))
            self._moleculeAtomPositions.append(numpy.asarray(atomPositions,numpy.float32))
            self._moleculeNames.append(moleculeName)
            self._moleculeAtomAminoAcides.append(numpy.asarray(atomAminoAcides,numpy.int32))
            self._moleculeAtomCharges.append(numpy.asarray(atomCharges,numpy.float32))
#             self._moleculeAtomBonds.append(numpy.asarray(atomBonds,numpy.int32))

        #print("Atom Names: ", atomsNames)
        #print("Atoms: ", atoms)
        #print("Atom Positions: ", atomPositions)
        #print("Amino Acids: ",atomAminoAcides)
        #print("Charges", atomCharges)
        print(moleculeName)
        #print("----------------")


        
if __name__ == '__main__':
    from time import time
    
    inputFileNameRec  = "../../deepbio-data/vinaoutput/comt/rec.pdbqt"
    inputFileNameMols = "../../deepbio-data/vinaoutput/comt/vina_out"
    
    initTime = time()
    ds = PDBQTDataset()
    
    print("Loading protein...")
    ds.load(inputFileNameRec)
    print("# molecules", len(ds._moleculeNames))

    print("Creating KDTress")
    ds.createMoleculeKDTrees()
    print("# molecule atom kdtrees:", len(ds._moleculeAtomKDTrees))
    
    i = 0
    for molName in ds._moleculeNames:
        print(molName, "# atoms:", len(ds._moleculeAtoms[i]))
        print("Atoms:")
        for atom in ds._moleculeAtoms[i]:
            print(ds.getTermByIndex(atom), end=' ')
        print()
        print("charges:")
        for v in ds._moleculeAtomCharges[i]:
            print("%.4f"%v, end=' ')
        print()
        print("amino:")
        for v in ds._moleculeAtomAminoAcides[i]:
            print(ds.getTermByIndex(v), end=' ')
        print()
        print("amino:")
        for v in ds._moleculeAtomPositions[i]:
            print(v, end=' ')
        print()

