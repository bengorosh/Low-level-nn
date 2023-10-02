#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//creating a simple nn that can learn XOR

#define numInputs 2
#define numOutputs 1
#define numHiddenNodes 2
#define numTrainingSets 4

double init_weights() { return ((double)rand()) / ((double)RAND_MAX); }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double dSigmoid(double x) { return x * (1 - x); }

void shuffle(int *array, size_t n) {
	if(n > 1) {
		size_t i;
		for(i = 1; i < n-1; i++) {
			size_t j = i + rand() / (RAND_MAX / (n-i) + 1);
			int t = array[j];
			array[j] = array[i];
			array[i] = t;
	}
}
}

int main(void) {

	srand(time(NULL));
	
	const double lr = 0.1;
	double hiddenLayer[numHiddenNodes];
	double outputLayer[numOutputs];

	double hiddenLayerBias[numHiddenNodes];
	double outputLayerBias[numOutputs];

	double hiddenWeights[numInputs][numHiddenNodes];
	double outputWeights[numHiddenNodes][numOutputs];

	double training_inputs[numTrainingSets][numInputs] =  {{0.0f, 0.0f},
														   {1.0f, 0.0f},
														   {0.0f, 1.0f},
														   {1.0f, 1.0f}};

	double training_outputs[numTrainingSets][numOutputs] =  {{0.0f},
															 {1.0f},
															 {0.0f},
															 {1.0f}};

    for(int i = 0; i < numInputs; i++) {
    	for(int j = 0; j < numHiddenNodes; j++) {
    		hiddenWeights[i][j] = init_weights();
    	}
    }

    for(int i = 0; i < numHiddenNodes; i++) {
    	for(int j = 0; j < numOutputs; j++) {
    		outputWeights[i][j] = init_weights();
    	}
    }


    for(int i = 0; i < numOutputs; i++) {
    		outputLayerBias[i] = init_weights();
    }

        for(int i = 0; i < numHiddenNodes; i++) {
    		hiddenLayerBias[i] = init_weights();
    }

    int trainingSetOrder[] = {0,1,2,3};

    int numberOfEpochs = 10000;

    //train nn for num of epochs

    for(int epoch = 0; epoch < numberOfEpochs; epoch++) {
    	shuffle(trainingSetOrder, numTrainingSets);
    	
    	for(int x = 0; x < numTrainingSets; x++){
    		int i = trainingSetOrder[x];
    		
    		//forward pass
    		//computes hidden layer activation

    		for(int j = 0; j <numHiddenNodes; j++) {
    			double activation = hiddenLayerBias[j];
    			
    			for(int k = 0; k < numInputs; k++) {
    				activation += training_inputs[i][k] * hiddenWeights[k][j];

    			}

    			hiddenLayer[j] = sigmoid(activation);
    		}

    		//computes output layer activation
    		
    		for(int j = 0; j <numOutputs; j++) {
    			double activation = outputLayerBias[j];
    			
    			for(int k = 0; k < numHiddenNodes; k++) {
    				activation += hiddenLayer[k] * outputWeights[k][j];

    			}

    			outputLayer[j] = sigmoid(activation);
    		}

    		//Print results from forward pass for each epoch

    		printf("Input: %g %g NN Output: %g Expected Output: %g \n",
    				training_inputs[i][0], training_inputs[i][1],
    				outputLayer[0], training_outputs[i][0]);

    		//Backward propagation of errors

    		//Compute change in output weights

    		double deltaOutput[numOutputs];

    		for(int j = 0; j < numOutputs; j++) {
    			double error = (training_outputs[i][j] - outputLayer[j]);
    			deltaOutput[j] = error * dSigmoid(outputLayer[j]);
    		} 

    		double deltaHidden[numHiddenNodes];

    		for(int j = 0; j < numHiddenNodes; j++) {
    			double error = 0.0f;
    			for(int k = 0; k < numOutputs; k++) {
    				error += deltaOutput[k] * outputWeights[j][k];
    			}
    			deltaHidden[j] = error * dSigmoid(hiddenLayer[j]);
    		}

    		//Apply changes in output weights

    		for(int j = 0; j < numOutputs; j++) {
    			outputLayerBias[j] += deltaOutput[j] * lr;
    			for(int k = 0; k < numHiddenNodes; k++) {
    				outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
    			}
    		}

    		//Apply changes in hidden weights

    		for(int j = 0; j < numHiddenNodes; j++) {
    			hiddenLayerBias[j] += deltaHidden[j] * lr;
    			for(int k = 0; k < numInputs; k++) {
    				hiddenWeights[k][j] += training_inputs[i][k] * deltaHidden[j] * lr;
    			}
    		}
    	}
    }

    // Print final weights after done training
    		fputs ( "Final Hidden Weights\n[ ", stdout);
    		for(int j = 0; j < numHiddenNodes; j++) {
    			fputs ("[ ", stdout);
    			for(int k = 0; k < numInputs; k++) {
    				printf ("%f ", hiddenWeights[k][j]);
    			}
    			fputs ("] ", stdout);
    		}

    		fputs ( "]\nFinal Hidden Biases\n[ ", stdout);
    		for(int j = 0; j < numHiddenNodes; j++) {
    			printf("%f ", hiddenLayerBias[j]);
    		}

    		fputs ( "Final Output Weights\n[ ", stdout);
    		for(int j = 0; j < numOutputs; j++) {
				fputs ("[ ", stdout);
    			for(int k = 0; k < numHiddenNodes; k++) {
    				printf ("%f ", outputWeights[k][j]);
    			}
    			fputs ("] ", stdout);
    		}

    		fputs ( "]\nFinal Output Biases\n[ ", stdout);
    		for(int j = 0; j < numOutputs; j++) {
    			printf("%f ", outputLayerBias[j]);
    		}

    		fputs ( "] \n", stdout);

    		return 0;
}