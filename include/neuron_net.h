#ifndef NEURON_NET_H
#define NEURON_NET_H

class NeuronNet {

private:

	int numberInputNeurons;
	int numberHiddenNeurons;
	int numberOutputNeurons;

	double *input;
	double *output;
	double *hiddenOutputs;
	double *outputsNet;
	double **Layer1_weights;
	double **Layer2_weights;


	void initializeWeights();
	void computeOutputs();
	void calculateGradientErrorFunction(double **gradientWeightsLayer1, double **gradientWeightsLayer2);
	void correctWeights(double **gradientWeightsLayer1, double **gradientWeightsLayer2, double learningRate);
	void BackWard(double learningRate,double**gradientWeightsLayer1, double**gradientWeightsLayer2);
	double calculateValueErrorFunction(double **trainData, double *trainLabel, int numberTrainImage);
	void setRandomOrder(int *order, int size);
	double hyperbolicTangent(double x);
	double deriviateHyperbolicTangent(double valueTanh);
	double *softmax(double *g, int numberNeurons);

public:

	NeuronNet(int _numberInputNeurons, int _numberHiddenNeurons, int _numberOutputNeurons);	
	~NeuronNet();

	double calculatePrecision(double **data, double *label, int numberImage);
	void trainNeuronNetwork(double **trainData, double *trainLabel, int numberTrainImage, int numberEpochs, double learningRate, double errorCrossEntropy);
};

#endif