
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include "neuron_net.h"
#include "math.h"

NeuronNet::NeuronNet(int numInput, int numHidden, int numOutput) {
	
	numberInputNeurons = numInput;
	numberHiddenNeurons = numHidden;
	numberOutputNeurons = numOutput;

	input = new double[numberInputNeurons];
	output = new double[numberOutputNeurons];

	hiddenOutputs = new double[numberHiddenNeurons];
	outputsNet = new double[numberOutputNeurons];

	srand(time(NULL));

	Layer1_weights = new double*[numberInputNeurons];
	for (int i = 0; i < numberInputNeurons; i++){
		Layer1_weights[i] = new double[numberHiddenNeurons];
		for (int j = 0; j < numberHiddenNeurons; j++)
			Layer1_weights[i][j] = (double(rand()) / (double)RAND_MAX) / 100.0;
	}
	Layer2_weights = new double* [numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++){
		Layer2_weights[s] = new double[numberOutputNeurons];
		for (int j = 0; j < numberOutputNeurons; j++)
			Layer2_weights[s][j] = (double(rand()) / (double)RAND_MAX) / 100.0;
	}
}

NeuronNet::~NeuronNet() {
	delete[] input;
	delete[] output;
	delete[] hiddenOutputs;
	delete[] outputsNet;

	for (int i = 0; i < numberInputNeurons; i++)
		delete[] Layer1_weights[i];
	delete[] Layer1_weights;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] Layer2_weights[i];
	delete[] Layer2_weights;
}

void NeuronNet::computeOutputs() {

	double *masOutputNeurons = new double[numberOutputNeurons];
	double *masHiddenNeurons = new double[numberHiddenNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		masHiddenNeurons[s] = 0;
		for (int i = 0; i < numberInputNeurons; i++) {
			masHiddenNeurons[s] += Layer1_weights[i][s] * input[i];
		}
		hiddenOutputs[s] = hyperbolicTangent(masHiddenNeurons[s]);
	}
	hiddenOutputs[0] = 1;

	for (int j = 0; j < numberOutputNeurons; j++) {
		masOutputNeurons[j] = 0;
		for (int s = 0; s < numberHiddenNeurons; s++) {
			masOutputNeurons[j] += Layer2_weights[s][j] * hiddenOutputs[s];
		}
	}

	outputsNet = softmax(masOutputNeurons, numberOutputNeurons);
	delete[] masHiddenNeurons;
	delete[] masOutputNeurons;
}

double NeuronNet::hyperbolicTangent(double x) {
	return tanh(x);
}

double* NeuronNet::softmax(double *g, int numberNeurons) {
	double* result = new double[numberNeurons];
	double scale = 0.0;

	for (int m = 0; m < numberNeurons; m++) {
		scale += exp(g[m]);
	}

	for (int j = 0; j < numberNeurons; j++) {
		result[j] = exp(g[j]) / scale;
	}

	return result;
}

void NeuronNet::calculateGradientErrorFunction(double **gradientWeightsLayer1, double **gradientWeightsLayer2) {
	double *sigmaLayer2 = new double[numberOutputNeurons];
	double *summa = new double[numberHiddenNeurons];
	double *derivativeActFuncHiddenLayer = new double[numberHiddenNeurons];

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			sigmaLayer2[j] = outputsNet[j] - output[j];
			gradientWeightsLayer2[s][j] = sigmaLayer2[j] * hiddenOutputs[s];
		}
	}
	
	for (int s = 0; s < numberHiddenNeurons; s++) {
		derivativeActFuncHiddenLayer[s] = (1 - hiddenOutputs[s]) * (1 + hiddenOutputs[s]);
	}

	for (int s = 0; s < numberHiddenNeurons; s++) {
		summa[s] = 0.0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			summa[s] += sigmaLayer2[j] * Layer2_weights[s][j];
		}
	}

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			gradientWeightsLayer1[i][s] = derivativeActFuncHiddenLayer[s] * summa[s] * input[i];
		}
	}

	delete[] sigmaLayer2;
	delete[] summa;
	delete[] derivativeActFuncHiddenLayer;
}

void NeuronNet::correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate) {

	for (int i = 0; i < numberInputNeurons; i++) {
		for (int s = 0; s < numberHiddenNeurons; s++) {
			double delta = learningRate * gradientWeightsLayer1[i][s];
			Layer1_weights[i][s] -= delta;
		}
	}

	for (int s = 0; s < numberHiddenNeurons; s++) {
		for (int j = 0; j < numberOutputNeurons; j++) {
			double delta = learningRate * gradientWeightsLayer2[s][j];
			Layer2_weights[s][j] -= delta;
		}
	}
}



void NeuronNet::BackWard(double learningRate, double**gradientWeightsLayer1, double**gradientWeightsLayer2) {
	computeOutputs();
	calculateGradientErrorFunction(gradientWeightsLayer1, gradientWeightsLayer2);
	correctWeights(gradientWeightsLayer1, gradientWeightsLayer2, learningRate);
}

double NeuronNet::calculateValueErrorFunction(double **trainData, double *trainLabel, int numberTrainImage) {
	double crossEntropy = 0;

	for (int image = 0; image < numberTrainImage; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			input[i] = trainData[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			output[j] = 0;
		}
		output[(int)trainLabel[image]] = 1;

		computeOutputs();

		for (int j = 0; j < numberOutputNeurons; j++) {
			crossEntropy += output[j] * log(outputsNet[j]);
		}
	}
	crossEntropy = -1 * crossEntropy / numberTrainImage;

	return crossEntropy;

}

void NeuronNet::setRandomOrder(int *order, int size) {
	int randomNumber, tmp;
	for (int i = 0; i < size; i++) {
		order[i] = i;
	}

	for (int i = 0; i < size; i++) {
		randomNumber = i + rand() % (size - i);
		tmp = order[i];
		order[i] = order[randomNumber];
		order[randomNumber] = tmp;
	}
}

void NeuronNet::trainNeuronNetwork(double **trainData, double *trainLabel, int numberTrainImage, int numberEpochs, double learningRate, double errorCrossEntropy) {
	
	double currentCrossEntropy = 0;
	int numberImage = 0;

	int *order = new int[numberTrainImage];

	double **gradientWeightsLayer1 = new double*[numberInputNeurons];
	double **gradientWeightsLayer2 = new double*[numberHiddenNeurons];

	for (int i = 0; i < numberInputNeurons; i++)
		gradientWeightsLayer1[i] = new double[numberHiddenNeurons];
	for (int s = 0; s < numberHiddenNeurons; s++)
		gradientWeightsLayer2[s] = new double[numberOutputNeurons];


	for (int epoch = 0; epoch < numberEpochs; epoch++) {
		printf("# epoch = %d \n", epoch);

		setRandomOrder(order, numberTrainImage);

		for (int image = 0; image < numberTrainImage; image++) {
			numberImage = order[image];
			for (int i = 0; i < numberInputNeurons; i++) {
				input[i] = trainData[numberImage][i];
			}

			for (int j = 0; j < numberOutputNeurons; j++) {
				output[j] = 0;
			}
			output[(int)trainLabel[numberImage]] = 1;

			BackWard(learningRate, gradientWeightsLayer1, gradientWeightsLayer2);
		}

		currentCrossEntropy = calculateValueErrorFunction(trainData, trainLabel, numberTrainImage);
		printf("    currentCrossEntropy = %f \n", currentCrossEntropy);
		
 		if (currentCrossEntropy < errorCrossEntropy) {
			break;
		}
	}

	delete[] order;
	for (int i = 0; i < numberInputNeurons; i++)
		delete[] gradientWeightsLayer1[i];
	delete[] gradientWeightsLayer1;

	for (int i = 0; i < numberHiddenNeurons; i++)
		delete[] gradientWeightsLayer2[i];
	delete[] gradientWeightsLayer2;
}

double NeuronNet::calculatePrecision(double **data, double *label, int numberImage) {
	double precision = 0;
	int truePositive = 0, falsePositive = 0;
	int maxIndex;

	for (int image = 0; image < numberImage; image++) {
		for (int i = 0; i < numberInputNeurons; i++) {
			input[i] = data[image][i];
		}

		for (int j = 0; j < numberOutputNeurons; j++) {
			output[j] = 0;
		}
		output[(int)label[image]] = 1;

		computeOutputs();

		maxIndex = 0;
		for (int j = 0; j < numberOutputNeurons; j++) {
			if (outputsNet[j] > outputsNet[maxIndex]) {
				maxIndex = j;
			}
		}		

		if (output[maxIndex] == 1.0) {
			truePositive++;
		}
		else {
			falsePositive++;
		}
	}

	precision = (double)truePositive / (double)(truePositive + falsePositive);

	return precision;
}




