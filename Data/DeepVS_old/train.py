'''
Created on May 6, 2018
@author: jana
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from data_processing import load_data, prepareMolData, prepareMinibatches, context_dictionary, loadProteinRestrictions
from model import DeepVS
from scorer_auc_enrichment_factor import Scorer
import numpy as np
import random
import sys
import os
from auc_scorer import AUCScorer
from collections import Counter
#from sklearn.datasets import make_classification
#from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
import itertools


class DeepVSExperiment:
    
    def __init__(self,
        embedding_size = 200,
        cf = 400,
        h  = 50,
        lr = 0.00001,
        kc = 6,
        kp = 2,
        num_epochs = 7,
        minibatchSize = 20,
        l2_reg_rate = 0.0001,
        use_Adam = True):

        self.embedding_size = embedding_size
        self.cf = cf
        self.h = h 
        self.lr = lr
        self.kc = kc
        self.kp = kp
        self.num_epochs = num_epochs 
        self.minibatchSize = minibatchSize
        self.l2_reg_rate = l2_reg_rate
        self.use_Adam = use_Adam
        
        self._aucSumByEpoch       = [0.0]*10
        self._EfMaxSumByEpoch     = [0.0]*10
        self._Ef2SumByEpoch       = [0.0]*10
        self._Ef20SumByEpoch      = [0.0]*10
        self._numProteinProcessed = 0
        #self._top3ByEpoch = []
        self._top3 = []
        self._testnames = []		

#    def confusion(self, prediction, truth, do_print = False):
#        confusion_vector = prediction / truth
#
#        true_positives = torch.sum(confusion_vector == 1).item()
#        false_positives = torch.sum(confusion_vector == float('inf')).item()
#        true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
#        false_negatives = torch.sum(confusion_vector == 0).item()
#        if do_print:
#            print("\t Actual class")
#            print("\t+(1)\t-(0)")
#            print("+(1)\t%i\t%i\tPredicted" % (true_positives,fpfalse_positives))
#            print("-(0)\t%i\t%i\tclass" % (false_negatives,true_negatives))
#
#        return true_positives, false_positives, true_negatives, false_negatives
	
    def run(self, datasetPath, proteinsNames_training, proteinsNames_test, proteinRestrictions):
        torch.manual_seed(31)   
        rng = random.Random(31)
        self.epoch = 0
        self._numProteinProcessed += 1.0
        top3ByEpoch = []


        testProRestrictions = proteinRestrictions.get(proteinsNames_test[0])
        
        if testProRestrictions is not None:
            i = len(proteinsNames_training) - 1
            while i > -1 :
                if proteinsNames_training[i] in testProRestrictions:
                    del proteinsNames_training[i]
                i -= 1
            
        # Preparing training dataset
        print("Loading data ...")
        molName_training, molClass_training, molData_training = load_data(datasetPath, self.kc, self.kp,
                                                                          proteinsNames_training, rng, randomize = True)

        # ---------------------------------------------------
        # Implementing oversampling to handle imbalanced data
        # ---------------------------------------------------
        #print("molData_training type: ", type(molData_training))
        #molData_training = np.reshape(molData_training, (-1, 1))
        #print("TYPES: ", type(molData_training), type(molData_training))
        #print("DATA SHAPE: ",molData_training.shape)
        #print("MOLDATA TRAINING LENGTH: ",len(molData_training))
        #print("proteinsNames_training: ", proteinsNames_training)
        #print("Preparing data ...")
		#----Oversampling training data
        #print("CLASS counter and LENGTH", Counter(molClass_training) ,
		#len(molClass_training))
        #ros = RandomOverSampler(sampling_strategy='minority', random_state=42, ratio = 4/5)
        #X_res, y_res = ros.fit_resample(molData_training,
        #molClass_training)
        #print('Resampled dataset class shape %s' % Counter(y_res))
		#----
        #print("X_res shape: ",X_res.shape)
        #print("X_res type: ", type(X_res))
        #X_res = np.reshape(X_res, (-1, ))
        #print("X_res reshaped: ",X_res.shape)
        #X_res = X_res.tolist()


        context_to_ix_training = context_dictionary(molData_training) # (X_res)
        molData_ix_training = prepareMolData(molData_training, context_to_ix_training) # (X_res, )
        molDataBatches_training = prepareMinibatches(molData_ix_training, molClass_training, self.minibatchSize) # ( , y_res, , )

        # Shuffle data so that all 1's will not appear at the end
        #rng.shuffle(molDataBatches_training)
		


#         print("CONTEXT_TO_IX_TRAINING FOR MOLDATA ", context_to_ix_training)
#         print("MOL DATA BATCHES FROM CONTEXT AND CLASS ", molDataBatches_training[:20])


        
        # Preparing test dataset
        molName_test, molClass_test, molData_test = load_data(datasetPath, self.kc, self.kp, proteinsNames_test,
                                                              rng, randomize = False)
        print("proteinsNames_test: ", proteinsNames_test)
        print("number of test molecules: ", len(molData_test))
        molData_ix_test = prepareMolData(molData_test, context_to_ix_training)
        molDataBatches_test = prepareMinibatches(molData_ix_test, molClass_test, self.minibatchSize)

#from collections import Counter
#from sklearn.datasets import make_classification
#from imblearn.over_sampling import RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
#X, y = make_classification(n_classes=2, class_sep=2,
#weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
#print('Original dataset shape %s' % Counter(y))
#ros = RandomOverSampler(random_state=42)
#X_res, y_res = ros.fit_resample(X, y)
#print('Resampled dataset shape %s' % Counter(y_res))		
		
#from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(random_state=0)
#X_resampled, y_resampled = ros.fit_resample(X, y)
#from collections import Counter
#print(sorted(Counter(y_resampled).items()))
        
        
        # Number of columns in the embedding matrix
        vocab_size = len(context_to_ix_training) 
        
        
        # Instantiate Model  Class
        model = DeepVS(vocab_size, self.embedding_size, self.cf, self.h, self.kc, self.kp)
        #print("---------VOCAB SIZE----------", vocab_size)
        #print("---------EMBEDDING size----------", self.embedding_size)
        #####################
        # Use GPU for model #
        #####################
        if torch.cuda.is_available():
            model.cuda()  
            print("using GPU!")
        
        
        # Instantiate Loss Class
        # This line was modified, original has no parameters
        loss_fuction = nn.NLLLoss(weight=torch.FloatTensor([1., 2.])) #weight=torch.FloatTensor([1., 5.])
        
        
        # Instantiate scorer
        scorer = Scorer()
        
        # AUC SCORER
        # aucscorer = AUCScorer()
        
        
        # Instantiate optimizer class: using Adam
        if self.use_Adam:
            optimizer  = optim.Adam(model.parameters(), self.lr, weight_decay = self.l2_reg_rate) #weight_decay = self.l2_reg_rate
            print('using Adam')            
        else:
            optimizer  = optim.SGD(model.parameters(), self.lr, weight_decay = self.l2_reg_rate)
            print('using SGD')
        
        print('lr = ', self.lr)
        print('ls_reg_rate = ', self.l2_reg_rate)
        
        
        

        # Train Model
        print("Training ...")
        for epoch in range(1, self.num_epochs+1):
            total_loss = 0.0
            model.train()
            predictions_cls = []
            truth = []
            for cmplx, cls, msk in molDataBatches_training:
                # convert contexts and classes into torch variables 
                if torch.cuda.is_available():
                    cls  = autograd.Variable(torch.LongTensor(cls).cuda())
                    cmplx = autograd.Variable(torch.LongTensor(cmplx).cuda())
                    mskv = autograd.Variable(torch.FloatTensor(msk).cuda())                  
                else:
                    cls  = autograd.Variable(torch.LongTensor(cls))
                    cmplx = autograd.Variable(torch.LongTensor(cmplx))
                    mskv = autograd.Variable(torch.FloatTensor(msk))
        
                model.zero_grad()
                                
                # Run the forwad pass 
                log_probs = model(cmplx, mskv)
                #print("CMPLX: ", cmplx[:5])
                #print("MSKV: ", mskv[:5])				
                
                # Compute loss and update model 
                loss = loss_fuction(log_probs,cls)
                #print("Probs: ", log_probs[:5])
                #print("Class: ", cls[:5])

                loss.backward()
                optimizer.step()
                total_loss += loss.data.cpu()        
            
            # shuffles the training set after each epoch
            rng.shuffle(molDataBatches_training)
            
            # sets model to eval (needed to use dropout in eval mode)
            model.eval()
            # Test model after each epoch
            correct = 0.0
            numberOfMolecules = 0.0
            total_loss_test = 0.0
            scores = []
            scr = []
            cls = []
            testMolId = 0
            nb_classes = 2
            conf_matrix = torch.zeros(nb_classes, nb_classes)
            for cmplx_test, cls_test, msk_test in molDataBatches_test:
               
                cls_test  = torch.LongTensor(cls_test)
                #print(cls_test)
                #print(cls_test)
                # convert contexts and classes into torch variables
                if torch.cuda.is_available():     
                    cls_test_v = autograd.Variable(cls_test.cuda())
                    cmplx_test = autograd.Variable(torch.LongTensor(cmplx_test).cuda())
                    mskv_test = autograd.Variable(torch.FloatTensor(msk_test).cuda())             
                else:
                    cls_test_v = autograd.Variable(cls_test)
                    cmplx_test = autograd.Variable(torch.LongTensor(cmplx_test))
                    mskv_test = autograd.Variable(torch.FloatTensor(msk_test)) 
                
                # Run the forwad pass 
                outputs = model(cmplx_test, mskv_test)
                loss_test = loss_fuction(outputs, cls_test_v)
     
                

                # Get predictions 
                

                # predict_mine = np.where(outputs.data >
                #  torch.LongTensor(0.3),
                #  torch.LongTensor(1), torch.LongTensor(0))


                # predict_mine = torch.where(outputs.data >
                # torch.LongTensor(0.3),
                # torch.LongTensor(1), torch.LongTensor(0))

                _, predicted = torch.max(outputs.data, 1)
                for cur_scr, cur_cls in zip(np.exp(outputs.data[:,1]), cls_test.cpu()):
                    scr.append(cur_scr)
                    cls.append(cur_cls)
                    scores.append([cur_scr, cur_cls, molName_test[testMolId]])
                    #print(cur_scr)
                    testMolId += 1
                numberOfMolecules += cls_test.size()[0]
                
                

                
                 
                if torch.cuda.is_available():
                    correct += (predicted.cpu() == cls_test.cpu()).sum()
                else:
                    correct += (predicted == cls_test).sum()
                    #print("If predicted: ", predicted)
                    #print("equals to test class: ",cls_test)
                    #print("Correct = ", correct)
                    #print("----------")      

                    for c, p in zip(cls_test.long(), predicted.long()):
                        conf_matrix[c,p] += 1
                
                total_loss_test += loss_test.data
            

            
                
#             print("Total correct: ", correct)
            accuracy = 100 * correct / numberOfMolecules
            print("--------------------------------------------------------------------------------------------")    
            print("epoch = %d;  total loss training = %.4f; total loss test = %.4f; accuracy = %f" %(epoch, total_loss/len(molDataBatches_training), total_loss_test/len(molDataBatches_test), accuracy))
            efAll, dataForROCCurve, efValues, aucValue, scoresToAuc, top3 = scorer.computeEnrichmentFactor_and_AUC(scores,removeRepetitions=True)
            aucScorer = AUCScorer(scoresToAuc)
            print("TOP 3 active ligands in epoch %d, "%epoch)
            print("1. name = %s prediction =  %.4f class = %d "% (top3[0][2],top3[0][0],top3[0][1]))
            print("2. name = %s prediction =  %.4f class = %d "% (top3[1][2],top3[1][0],top3[1][1]))
            print("3. name = %s prediction =  %.4f class = %d "% (top3[2][2],top3[2][0],top3[2][1]))
            pos_0_scr= [x[0] for x in scores]
            pos_1_cls= [x[1] for x in scores]
            print(scores[:20])
            print("----Pytorch Confusion Matrix----")
            print(conf_matrix)
			
			
			# -------
            # Confusion matrix
            # -------

            #print("--------------Confusion matrix from DeepVS threshold = 0.5--------------")
            aucScorer.confusion_matrix(threshold=0.5,do_print=False) 

            
            # -------------------------------
            # 			ROC and AUC
            # -------------------------------
			# calculate AUC
            auc_scr_roc = roc_auc_score(pos_1_cls, pos_0_scr)
            print('AUC with roc_auc_score(): %.3f' % auc_scr_roc)
#			# calculate roc curve
#            fpr, tpr, thresholds = roc_curve(pos_1_cls, pos_0_scr)
#            # plot no skill
#            plt.plot([0, 1], [0, 1], linestyle='--')
#            # plot the roc curve for the model
#            plt.plot(fpr, tpr, marker='.')
#            # show the plot
#            plt.show()
			
			# ------------------------------
            # Precision and recall curve and AUC
            # -------------------------------
			# predict class values
            #yhat = model.predict(testX)
            #calculate precision-recall curve
            precision, recall, thresholds = precision_recall_curve(pos_1_cls, pos_0_scr)
            # calculate F1 score
            #f1 = f1_score(testy, yhat)
            # calculate precision-recall AUC
            auc_scr = auc(recall, precision)
            # calculate average precision score
            ap = average_precision_score(pos_1_cls, pos_0_scr) 
            print('Precision and recall auc=%.3f ap=%.3f' % (auc_scr, ap))
			#print("Precision AUC and average precision score: ", auc)
            # plot no skill 
#            plt.plot([0, 1], [0.1, 0.1], linestyle='--')
#            # plot the precision-recall curve for the model
#            plt.plot(recall, precision, marker='.')
#            # show the plot
#            plt.show()




            self._aucSumByEpoch[self.epoch]    += auc_scr_roc #aucValue # auc_score
            self._EfMaxSumByEpoch[self.epoch]  += efValues[2]
            self._Ef2SumByEpoch[self.epoch]    += efValues[0]
            self._Ef20SumByEpoch[self.epoch]   += efValues[1]
            top3ByEpoch.append(top3)
            
            self.epoch += 1
            
        
            
            
        allEpochsTop3 = []
		#self._top3ByEpoch.copy()
        print("Average AUC, EF2, EF20, EFMax by epoch for %d proteins:"%self._numProteinProcessed)
        for k in range(self.num_epochs):
            print("Ep: %d, AUC: %.4f -"%(k+1, self._aucSumByEpoch[k]/self._numProteinProcessed), end=' ')
        print(" ")
        for k in range(self.num_epochs):
            print("Ep: %d, EF 2%%: %.4f -"%(k+1, self._Ef2SumByEpoch[k]/self._numProteinProcessed), end=' ')
        print(" ")
        for k in range(self.num_epochs):
            print("Ep: %d, EF 20%%: %.4f -"%(k+1, self._Ef20SumByEpoch[k]/self._numProteinProcessed), end=' ')
        print(" ")
        for k in range(self.num_epochs):
            print("Ep: %d, EF Max: %.4f -"%(k+1, self._EfMaxSumByEpoch[k]/self._numProteinProcessed), end=' ')
        print(" ")
        print("TOP 3 ligands for protein: %s by epoch"%proteinNames_test)
        for k in range(self.num_epochs):
            print("Top 3 for Epoch: %d"%(k+1))
            top3Epoch = top3ByEpoch[k]
            print("1. name = %s prediction =  %.4f class = %d "% (top3Epoch[0][2],top3Epoch[0][0],top3Epoch[0][1]))
            print("2. name = %s prediction =  %.4f class = %d "% (top3Epoch[1][2],top3Epoch[1][0],top3Epoch[1][1]))  
            print("3. name = %s prediction =  %.4f class = %d "% (top3Epoch[2][2],top3Epoch[2][0],top3Epoch[2][1]))
        print(" ")
        sys.stdout.flush()
        #item[1] for item in top3Epoch[2]
        #allEpochsTop3.sort(reverse=True)
        for item in top3ByEpoch:
            for i in item:
                allEpochsTop3.append(i)
        allEpochsTop3.sort(reverse=True)
        #print("ALL :", allEpochsTop3)

        valid1 = False
        valid2 = False
        while(not valid1):
            if(allEpochsTop3[0][2] == allEpochsTop3[1][2]):
                del allEpochsTop3[1]
            else: 
                valid1 = True
        while(not valid2):
            if(allEpochsTop3[1][2] == allEpochsTop3[2][2]):
                del allEpochsTop3[2]
            else:
                valid2 = True
        self._top3.append(allEpochsTop3[:3])
        self._testnames.append(proteinNames_test)
		
					
					
        
		
if __name__ == '__main__':

    '''
    Definition of Hyperparameters:
    
    embedding_size = embedding size of d^atm, d^amino, d^chg, d^dist
    cf = number of convolutional filters
    h =  number of hidden units
    lr = learning rate
    kc = number of neighboring atoms from compound
    kp = number of neighboring atoms from protein
    num_epoch = number of epochs
    '''

    
    dvsExp = DeepVSExperiment(
        embedding_size = 200,
        cf = 400,
        h  = 50,
        lr = 0.00001,
        kc = 6,
        kp = 2,
        num_epochs = 7,
        minibatchSize = 20,
        l2_reg_rate = 0.0001,
        use_Adam = True)
    
    
    
    proteinNames = ['3tfq']
#    proteinNames = ['3klm', '3tfq', '1a4g', '1a5h', '1adw', '1ah3' #
#                    ,'1b8o', '1ckp', '1cx2', '1e3g', '1eve', '1f0r', '1fm9', '1hw8', '1i00']
#                     ,'1j8f', '1m17', '1nhz', '1ouk', '1p44', '1r4l', '1s3v', '1s6p', '1uou'
#                     ,'1uy6', '1uze', '1w4r', '1xjd', '1xoi', '1xp0', '1z11', '2afx', '2b1p'
#                     ,'2dg3', '2iwi', '2oo8', '2p1t', '2p54', '2src', '2vgo', '2vwz', '2w31'
#                     ,'2w8y', '2wcg', '2xch', '2z94', '3bc3', '3c7q', '3dbs', '3dds', '3elj'
#                     ,'3eml', '3ewj', '3fdn', '3frg', '3hng', '3i4b', '3k5e', '3kc3', '3kk6'
#                     ,'3kx1', '3l3m', '3lbk', '3lxl', '3max', '3mhw', '3mj1', '3mpm', '3npc'
#                     ,'3nu3', '3nw7', '3ny9', '3oll', '3pp0', '3qkl', '3r04', '3rm2', '3sff'
#                     ,'3skc', '3v8s']
    
    
                    #['3tfq', '1a5h']
                    # ace
                    #'ache', 'ada', 'alr2', 'ampc', 'ar', 'cdk2','comt', 'cox1', 
                    #'cox2', 'dhfr', 'egfr', 'er_agonist', 'er_antagonist', 
                    #'fgfr1', 'fxa', 'gart', 'gpb', 'gr', 'hivpr', 'hivrt', 'hmga', 'hsp90', 'inha', 'mr', 
                    #'na', 'p38', 'parp', 'pde5', 'pdgfrb', 'pnp', 'ppar', 'pr', 'rxr', 'sahh', 
                    #'src', 'thrombin', 'tk', 'trypsin', 'vegfr2']
    
    
 
    datasetPath  = 'dud_vinaout_deepvs/'
    proteinGroupsFileName = 'protein.groups' 
    proteinCrossEnrichmentFileName = 'protein.cross_enrichment'
    
    
    proteinNames = [x for x in proteinNames if os.path.isfile(datasetPath+x+".deepvs")]
#     print(len(filtered))
#     for i, pr in enumerate(proteinNames):
# #         if(str(prot).__contains__("1hw8")):
# #             print("This: ",prot)
# #         else:
#         print(i, pr)
  
#         if(not os.path.isfile(datasetPath+pr.strip()+".deepvs")):
#             #print("File does not exist in directory: ", datasetPath+prot+".deepvs")
#            # print("Removing it from the list...")
#             proteinNames.remove(pr)
#     print(count)
    print("New protein Names: ", proteinNames)
    print("Length of protein list", len(proteinNames))

    proteinRestrictions = loadProteinRestrictions(proteinGroupsFileName,proteinCrossEnrichmentFileName)
    
    #top3AllTime = []
    for pName in proteinNames:
        proteinNames_test = []
        proteinNames_training = ''
        if pName in proteinNames:
            proteinNames_test.append(pName)
            proteinNames_training = proteinNames[:]
            del proteinNames_training[proteinNames_training.index(pName)]
            
            print("======================================================================")
            print("Experimental results for protein:", proteinNames_test)
            print("======================================================================")
            dvsExp.run(datasetPath, proteinNames_training, proteinNames_test, proteinRestrictions)
			
    for protName, ligands in zip(dvsExp._testnames, dvsExp._top3):
        print("TOP 3 highest predicted active ligands for protein: %s"%protName)
        print("1. name = %s prediction =  %.4f class = %d "% (ligands[0][2],ligands[0][0],ligands[0][1]))
        print("2. name = %s prediction =  %.4f class = %d "% (ligands[1][2],ligands[1][0],ligands[1][1]))  
        print("3. name = %s prediction =  %.4f class = %d "% (ligands[2][2],ligands[2][0],ligands[2][1]))
