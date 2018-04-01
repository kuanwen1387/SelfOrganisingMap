/********************************************************
*							*
*	CSCI464 Assignment 3				*
*	Student Name: Kuan Wen Ng			*
*	Student Number: 5078052				*
*	Email : kwn961@uowmail.edu.au			*
*	Filename: som.cpp				*
*	Description: Self Organizing Map		*
*							*
********************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <utility>
#include <fstream>
using namespace std;

typedef vector<int> SampleIndex;
typedef vector<double> WeightVector;
typedef vector<WeightVector> Dataset;
typedef vector<vector<WeightVector> > Map;
typedef pair<int, int> BestIndex;

class SOM
{
private:
	Dataset dataset;	//Dataset
	Map map;	//2d lattice
	int neuronRow;	//Lattice height
	int neuronCol;	//Lattice width
	int sampleSize;	//Dataset size
	int inputDim;	//Sample dimension
	int ordering;	//Epoch for ordering phase
	int epoch;	//Epoch
	SampleIndex sampleIndex;	//Index for shuffling
	double learningRate;	//Learning rate
	double mapRadius;	//Initial map radius
	double timeConst;	//Time constant

public:
	SOM();
	~SOM();
	void readData(char * );
	void shuffle();
	void initWeights();
	void train();
	double vecDistance(const WeightVector , const WeightVector );
	BestIndex getBest(const WeightVector );
	double criterion(const Map );
	void saveWeights(char *);
	double randDouble();
	int randInt();
};

SOM::SOM()
{
	neuronRow = 10;
	neuronCol = 10;
	ordering = 1000;
	epoch = 2000;
	learningRate = 0.1;
	mapRadius = sqrt(pow((neuronRow / 2), 2.0) + pow((neuronCol / 2), 2.0));
	timeConst = ordering / log(mapRadius);
}

SOM::~SOM()
{
}

//Function to read dataset
void SOM::readData(char *filename)
{
	ifstream inputFile;
	char readString[10000];
	string tempString;
	int size = 1, readIndex = 0;

	sampleIndex.push_back(size - 1);
	inputFile.open(filename);

	//Read data from file
	if (inputFile.good() && !inputFile.eof())
	{
		//Get size of dataset
		inputFile.getline(readString, 10000, '\n');

		for (int index = 0; index < strlen(readString); index++)
		{
			if (readString[index] == ' ')
			{
				size++;
				sampleIndex.push_back(size - 1);
			}
		}

		sampleSize = size;
		inputFile.seekg(0, ios::beg);
		dataset.resize(sampleSize);

		//Read each sample
		while (inputFile.good() && !inputFile.eof())
		{
			if (readIndex == sampleSize)
				readIndex = 0;

			if (inputFile >> tempString)
			{
				dataset[readIndex].push_back(atof(tempString.c_str()));
				readIndex++;
			}
		}
		inputDim = dataset[0].size();

		inputFile.close();
	}
}

//Function for shuffling dataset
void SOM::shuffle()
{
	SampleIndex tempSampleIndex = sampleIndex;
	int tempIndex;

	sampleIndex.clear();

	for (int index = sampleSize - 1; index >= 0; index--)
	{
		tempIndex = (rand() % (index + 1));
		sampleIndex.push_back(tempSampleIndex[tempIndex]);
		tempSampleIndex.erase(tempSampleIndex.begin() + tempIndex);
	}
}

//Function to initialize weights of neurons
void SOM::initWeights()
{
	map.resize(neuronRow);

	for (int rowIndex = 0; rowIndex < neuronRow; rowIndex++)
	{
		map[rowIndex].resize(neuronCol);

		for (int colIndex = 0; colIndex < neuronCol; colIndex++)
		{
			map[rowIndex][colIndex].resize(inputDim, 0);

			for (int dimIndex = 0; dimIndex < inputDim; dimIndex++)
				map[rowIndex][colIndex][dimIndex] = randDouble();
		}
	}
}

//Function to train the feature map
void SOM::train()
{
	double cLearningRate, nbRadius, nbRadiusSq, influence, neuronDistance;
	BestIndex bestIndex;	//Index for Best Matching Neuron
	Map cMap;	//Feature map for previous epoch
	char filename1[] = "initial.txt", filename2[] = "middle.txt", filename3[] = "converge.txt";

	//Save feature map at initial state
	saveWeights(filename1);

	//Training
	for (int cEpoch = 0; cEpoch < epoch; cEpoch++)
	{
		cLearningRate = max(learningRate * exp(-cEpoch / ordering), 0.01);	//Update learning rate
		nbRadius = mapRadius * exp(-cEpoch / timeConst);	//Update neighbour radius
		nbRadiusSq = nbRadius * nbRadius;
		cMap = map;	//Store feature map of previous epoch
		shuffle();	//Shuffle samples

		//Train with each sample from dataset
		for (int cSampleIndex = 0; cSampleIndex < sampleSize; cSampleIndex++)
		{
			//Get best matching unit
			bestIndex = getBest(dataset[sampleIndex[cSampleIndex]]);

			//Update each neurons in lattice
			for (int rowIndex = 0; rowIndex < neuronRow; rowIndex++)
			{
				for (int colIndex = 0; colIndex < neuronCol; colIndex++)
				{
					//Compute euclidean distance of neuron from BMU
					neuronDistance = pow(double(bestIndex.first - rowIndex), 2.0) + pow(double(bestIndex.second - colIndex), 2.0);

					//Update neuron if within distance
					if (neuronDistance < nbRadiusSq)
					{
						//Update magnitude
						influence = exp(-neuronDistance / (2 * nbRadiusSq));

						//Update each weight of neuron
						for (int weightIndex = 0; weightIndex < inputDim; weightIndex++)
							map[rowIndex][colIndex][weightIndex] += influence * cLearningRate
							* (dataset[sampleIndex[cSampleIndex]][weightIndex] - map[rowIndex][colIndex][weightIndex]);
					}
				}
			}
		}

		//Output average distance of each neuron from previous epoch
		cout << "Epoch " << cEpoch + 1 << " Criterion: " << criterion(cMap) << endl;

		//Save feature map at middle state
		if (cEpoch == epoch / 2)
			saveWeights(filename2);

		//Save feature map at end state
		if (cEpoch == epoch - 1)
			saveWeights(filename3);
	}
}

//Function to compute euclidean distance of neurons
double SOM::vecDistance(const WeightVector sample, const WeightVector neuron)
{
	double distance = 0;

	for (int index = 0; index < inputDim; index++)
	{
		distance += (sample[index] - neuron[index]) * (sample[index] - neuron[index]);
	}

	return sqrt(distance);
}

//Function to get best matching neuron
BestIndex SOM::getBest(const WeightVector sample)
{
	BestIndex bestIndex;
	double bestDistance = sqrt(inputDim * 2 * 2), tempDistance;

	//Search all neurons in lattice
	for (int rowIndex = 0; rowIndex < neuronRow; rowIndex++)
	{
		for (int colIndex = 0; colIndex < neuronCol; colIndex++)
		{
			//Compute euclidean distance between neurons
			tempDistance = vecDistance(sample, map[rowIndex][colIndex]);

			//Store smallest distance and index
			if (tempDistance < bestDistance)
			{
				bestDistance = tempDistance;
				bestIndex.first = rowIndex;
				bestIndex.second = colIndex;
			}
		}
	}

	return bestIndex;
}

//Function to compute average distance of neurons between epochs
double SOM::criterion(const Map cMap)
{
	double sumDistance = 0;

	for (int rowIndex = 0; rowIndex < neuronRow; rowIndex++)
	{
		for (int colIndex = 0; colIndex < neuronCol; colIndex++)
			sumDistance += vecDistance(map[rowIndex][colIndex], cMap[rowIndex][colIndex]);
	}
	return sumDistance / (neuronRow * neuronCol);
}

//Function to save feature map to file
void SOM::saveWeights(char *filename)
{
	ofstream outputFile;

	outputFile.open(filename);

	for (int rowIndex = 0; rowIndex < neuronRow; rowIndex++)
	{
		for (int colIndex = 0; colIndex < neuronCol; colIndex++)
		{
			for (int weightIndex = 0; weightIndex < inputDim; weightIndex++)
				outputFile << int(map[rowIndex][colIndex][weightIndex] * 255) << " ";

			outputFile << endl;
		}
	}

	outputFile.close();
}

//Random number 0 to 1
double SOM::randDouble()
{
	return (rand() / (double)(RAND_MAX));
}

int main()
{
	char filename[] = "SOM_MNIST_data.txt";
	SOM som;

	srand(time(0));	//Randomize weights
	som.readData(filename);	//Read dataset from file
	som.initWeights();	//Initialize weights of neurons
	som.train();	//Train feature map

	return 0;
}
