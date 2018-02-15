#ifndef READ_MNIST_H
#define READ_MNIST_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <math.h>

using namespace std;

int reverseInt(int i) {
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}


void readSetImage(char *fileName, double **data) {
	ifstream file(fileName, ios::binary);
	if (file.is_open()) {
		int magicNumber = 0;
		int numberImages = 0;
		int numberRows = 0;
		int numberCols = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = reverseInt(magicNumber);

		if(magicNumber != 2051) {
			printf("\nInvalid MNIST image file! \n");
			exit(1);
		}

		file.read((char*)&numberImages, sizeof(int));
		numberImages = reverseInt(numberImages);

		file.read((char*)&numberRows, sizeof(int));
		numberRows = reverseInt(numberRows);

		file.read((char*)&numberCols, sizeof(int));
		numberCols = reverseInt(numberCols);

		for (int i = 0; i < numberImages; i++) {
			data[i][0] = 1;
			int k = 1;
			for (int r = 0; r < numberRows; r++) {
				for (int c = 0; c < numberCols; c++) {
					unsigned char pixel = 0;
					file.read((char*)&pixel, sizeof(unsigned char));
					data[i][k] = (double)pixel/255.0;
					k++;
				}
			}
		}
	}
	else {
		printf("\nError opening file! \n");
		exit(1);
	}
}


void readSetLabel(char *filename, double *label) {
	ifstream file(filename, ios::binary);
	if (file.is_open()) {
		int magicNumber = 0;
		int numberItems = 0;

		file.read((char*)&magicNumber, sizeof(int));
		magicNumber = reverseInt(magicNumber);

		if(magicNumber != 2049) {
			printf("Invalid MNIST label file! \n");
			exit(1);
		}
			
		file.read((char*)&numberItems, sizeof(int));
		numberItems = reverseInt(numberItems);
		for (int i = 0; i < numberItems; i++) {
			unsigned char l = 0;
			file.read((char*)&l, sizeof(unsigned char));
			label[i] = (double)l;
		}
	}
	else {
		printf("\nError opening file! \n");
		exit(1);
	}
}

void printImage(double **data, int number, int row, int col) {
	FILE *fout;
	fout = fopen("ImageMNIST.txt", "w");
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < col; c++) {
			fprintf(fout, "%lf ", data[number][1 + r*row + c]);
		}
		fprintf(fout, "\n");
	}
	fclose(fout);
}

#endif