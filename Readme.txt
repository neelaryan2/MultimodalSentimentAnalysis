Title - Multimodal Sentiment Analysis : Case studies and Paradigms

Team Members :
	1. Shreya Pathak 	- 180050100
	2. Mohammad Ali Rehan 	- 180050061
	3. Neel Aryan Gupta 	- 180050067

Multimodal Sentiment Analysis includes taking input as feature vectors of audio, video and text of a speaking person and figuring out the emotion he/she is expressing.
We present models that give good results by systematic integration of information across time and modalities. The models differ in the type of data they need, based on their alignment

Hierarchical Fusion for word aligned data: The model first fuses the modalities in pairs by passing them through a GRU and taking their weighted sum followed by tanh. This process is again repeated at the next fusion layer.
Finally we apply a GRU on the trimodal fused vectors and get our final representation of the utterances.

Memory Fusion Network for label aligned data: The system of LSTMs helps in generating memories by exploiting temporal relationship between inputs. DMAN learns how modalities affect each other. It studies change in elements of memory vector in successive timesteps and suppresses those that remain constant. MGM is like an LSTM but here the gates are neural networks. It  learns the time dependent relation between the cross modal memories (ĉ). We concatenate the final output of LSTM and MGM and use this to predict the sentiment.


How to run the code?
	
1. Hierarchical Fusion Network
-Trimodal

	-IEMOCAP
		- To run trimodal hfusion for iemocap upload the script 'trimodal-fusion-for-iemocap.py' to kaggle 

		- Import the dataset from:  https://www.kaggle.com/alirehan/multimodal-sentiment

		- Run the script
	-MOSEI
		- To run trimodal hfusion  for iemocap upload the script 'trimodal-fusion-for-mosei.py' to kaggle 

		- Import the dataset from:  https://www.kaggle.com/alirehan/mosei-dataset-3way

		- Run the script



2. Memory Fusion Network (MFN)

- Data for MFN is available at : https://www.kaggle.com/neelaryan/multimodal-sentiment-analysis

- Only 1 dataset (small in size) youtube_data.pkl has been included for running purposes. Other (large) datasets are available at previously specified kaggle link.

- memory_fusion_network.py contains the keras code for model architecture only and utils_nlp.py contains utility functions. Both these scripts are imported into mfn_keras.py and mfn_kfold.py which are used to train the model.

- mfn_keras.py is used to train a single MFN model and generate confusion matrix, classification report and tag scores. This script requires the following optional arguments :
  --data DATA           		the path to the dataset to use for training and testing.
  --mode {T,A,V,TA,TV,AV,TAV}	the combination of modality to use for training.
  --mem MEM             		the hyperparameter memory size for multiview gated memory.
  For example, this command will train a bimodal MFN model:
  python3 mfn_keras.py --data youtube_data.pkl --mode TA -- mem 256
  T, A and V stand for Text, Audio and Video (modalities) respectively.

- mfn_kfold.py is used to perform kFold validation and generate the k-averaged confusion matrix, classification report and tag scores. This script requires the following optional arguments :
  --data DATA           		the path to the dataset to use for training and testing.
  --mode {T,A,V,TA,TV,AV,TAV}	the combination of modality to use for training.
  --mem MEM             		the hyperparameter memory size for multiview gated memory.
  --k K                 		k-Fold validation.
  For example, this command will train 5 trimodal MFN models and show their averaged results:
  python3 mfn_kfold.py --data youtube_data.pkl --mode TAV -- mem 512 --k 5

- On running any of the above 2 scripts, the generated results as well as the model weights would be saved in the working directory.

- For further analysis, kaggle notebooks (trainable on CPU and GPU) involving MFN are also available at :
	- https://www.kaggle.com/neelaryan/mfn-keras
	- https://www.kaggle.com/neelaryan/mfn-kfold
	- https://www.kaggle.com/neelaryan/utils-nlp
	- https://www.kaggle.com/neelaryan/memory-fusion-network
	- https://www.kaggle.com/neelaryan/cmu-mosi-explore
  
