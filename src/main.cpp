
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include "neuron_net.h"
#include "read_mnist.h"



struct TaskConfig{
	char  *fileTrainImageMNIST;
	char  *fileTrainLabelsMNIST;
	char  *fileTestImageMNIST;
	char  *fileTestLabelsMNIST;
	int    numberHiddenNeurons;
	int    numberEpochs;
	double learningRate;
	double errorCrossEntropy;

	TaskConfig()
	{
		fileTrainImageMNIST = new char[500];
		fileTrainLabelsMNIST = new char[500];
		fileTestImageMNIST = new char[500];
		fileTestLabelsMNIST = new char[500];

		numberHiddenNeurons = 100;
		numberEpochs = 10;
		learningRate = 0.008;
		errorCrossEntropy = 0.005;
	}
};


void read_config();

TaskConfig task;

void output_setting()
{
	printf("############# parameters task #############\n");
	printf("1: Path to MNIST train-images = %s \n", task.fileTrainImageMNIST);
	printf("2: Path to MNIST train-labels = %s \n", task.fileTrainLabelsMNIST);
	printf("3: Path to MNIST test-images = %s \n", task.fileTestImageMNIST);
	printf("4: Path to MNIST test-labels  = %s \n", task.fileTestLabelsMNIST);
	printf("5: number hidden neuron (default = 200)  %d \n", task.numberHiddenNeurons);
	printf("6: maxEpochs (default = 50) %d \n", task.numberEpochs);
	printf("7: learnRate (default = 0.008) %f \n", task.learningRate);
	printf("8: crossError stop in train (default 0.005) %f \n", task.errorCrossEntropy);
}

void set_path(char *s, char *value)
{
	if (strcmp(s, "fileTrainImageMNIST,") == 0)
	{
		strcpy(task.fileTrainImageMNIST, value);
	}
	if (strcmp(s, "fileTrainLabelsMNIST,") == 0)
	{
		strcpy(task.fileTrainLabelsMNIST, value);
	}
	if (strcmp(s, "fileTestImageMNIST,") == 0)
	{
		strcpy(task.fileTestImageMNIST, value);
	}
	if (strcmp(s, "fileTestLabelsMNIST,") == 0)
	{
		strcpy(task.fileTestLabelsMNIST, value);
	}
}

void set_param(char *s, double value)
{
	if (strcmp(s, "numberHiddenNeurons,") == 0)
	{
		task.numberHiddenNeurons = (int)value;
	}
	if (strcmp(s, "numberEpochs,") == 0)
	{
		task.numberEpochs = (int)value;
	}
	if (strcmp(s, "learningRate,") == 0)
	{
		task.learningRate = value;
	}
	if (strcmp(s, "errorCrossEntropy,") == 0)
	{
		task.errorCrossEntropy = value;
	}
}

void read_file_conf(FILE *f_conf)
{
	char s[500];
	char val_str[500];
	double val = 0;
	int cnt = 0;
	int k = 0;
	while (!feof(f_conf))
	{
		cnt = fscanf(f_conf, "%s", s);
		k++;
		if ((cnt > 0) && (k <= 4))
		{
			fscanf(f_conf, "%s", val_str);
			set_path(s, val_str);
		}

		if ((cnt > 0) && (k > 4))
		{
			fscanf(f_conf, "%lf", &val);
			set_param(s, val);
		}

	}

	fclose(f_conf);
}

void read_config()
{
	FILE * f_conf = fopen("config.txt", "r");
	if (f_conf == NULL)
	{
		printf("Use default setting \n");
	}
	else
	{
		read_file_conf(f_conf);
	}

	output_setting();
}
int main(int argc, char* argv[])
{
	read_config();
	
	char *fileTrainImageMNIST = task.fileTrainImageMNIST;
	char *fileTrainLabelsMNIST = task.fileTrainLabelsMNIST;
	char *fileTestImageMNIST = task.fileTestImageMNIST;
	char *fileTestLabelsMNIST = task.fileTestLabelsMNIST;
	int numberHidden = task.numberHiddenNeurons + 1;
	int numberEpochs = task.numberEpochs;
	double learningRate = task.learningRate;
	double errorCrossEntropy = task.errorCrossEntropy;

	int width = 28, height = 28;
	int numberTrainImage = 60000;
	int numberTestImage = 10000;

	int numberInput = width * height + 1;
	int numberOutput = 10;

	double **trainData = new double*[numberTrainImage];
	for (int i = 0; i < numberTrainImage; i++)
		trainData[i] = new double[numberInput];
	readSetImage(fileTrainImageMNIST, trainData);

	double *trainLabel = new double[numberTrainImage];
	readSetLabel(fileTrainLabelsMNIST, trainLabel);

	double **testData = new double*[numberTestImage];
	for (int i = 0; i < numberTestImage; i++)
		testData[i] = new double[numberInput];
	readSetImage(fileTestImageMNIST, testData);

	double *testLabel = new double[numberTestImage];
	readSetLabel(fileTestLabelsMNIST, testLabel);

	printf("\n Run training algorithm ... \n \n");
	NeuronNet network = NeuronNet(numberInput, numberHidden, numberOutput); 
	network.trainNeuronNetwork(trainData, trainLabel, numberTrainImage, numberEpochs, learningRate, errorCrossEntropy);
	double precision = network.calculatePrecision(trainData, trainLabel, numberTrainImage);
	printf("precision train = %f \n", precision);

	precision = network.calculatePrecision(testData, testLabel, numberTestImage);
	printf("precision test = %f \n", precision);

	for (int i = 0; i < numberTrainImage; i++)
		delete[] trainData[i];
	delete[] trainData;

	for (int i = 0; i < numberTestImage; i++)
		delete[] testData[i];
	delete[] testData;

	delete[] trainLabel;
	delete[] testLabel;
	
	printf("runtime = %f min \n", (clock()/1000.0) / 60);
	system("pause");
	return 0;
}