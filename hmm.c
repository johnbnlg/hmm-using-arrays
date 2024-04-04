#include "hmm.h"

int getMaxSequenceLength(SequencesSet set){
    int i, maxLength = 0;
    for (i = 0; i < set.count; i++) {
        maxLength = set.sequence[i].length > maxLength ? set.sequence[i].length : maxLength;
    }
    return maxLength;
}

int sequencesSetBufferSize(SequencesSet set) {
    int i;
    int size = 1; /** sequences count*/
    for (i = 0; i < set.count; i++) {
        size += set.sequence[i].length + 1; /** sequence length + data length*/
    }
    return size;
}

void sequencesSetToBuffer(SequencesSet set, int buffer[]) {
    int i, *lengthRunner = buffer + 1, *dataRunner = buffer + set.count + 1;
    buffer[0] = set.count;
    for (i = 0; i < set.count; i++) {
        *lengthRunner = set.sequence[i].length;
        lengthRunner++;

        memcpy(dataRunner, set.sequence[i].data, set.sequence[i].length * sizeof(int));
        dataRunner += set.sequence[i].length;
    }
}

SequencesSet sequencesSetFromBuffer(int buffer[]) {
    int i, *lengthRunner = buffer + 1, *dataRunner = buffer + buffer[0] + 1;
    SequencesSet set;
    set.count = buffer[0];
    for (i = 0; i < set.count; i++) {
        set.sequence[i].length = *lengthRunner;
        lengthRunner++;

        memcpy(set.sequence[i].data, dataRunner, set.sequence[i].length * sizeof(int));
        dataRunner += set.sequence[i].length;
    }
    return set;
}

void sequencesSetSplit(SequencesSet set, int subsetsCount, SequencesSet subsets[]) {
    int i, j, chunkSize = set.count / subsetsCount, remainder = set.count % subsetsCount;
    int currentSequenceIndex = 0;
    for (i = 0; i < subsetsCount; ++i) {
        SequencesSet currentSubset = subsets[i];
        currentSubset.count = chunkSize;
        if (remainder > 0) {
            currentSubset.count++;
            remainder--;
        }
        for (j = 0; j < currentSubset.count; j++) {
            Sequence currentSequence = set.sequence[currentSequenceIndex];
            currentSubset.sequence[j].length = currentSequence.length;
            memcpy(currentSubset.sequence[j].data, currentSequence.data, currentSequence.length * sizeof(int));
            currentSequenceIndex++;
        }
    }
}

Hmm hmmFromMarkovChainSet(MarkovChainsSet chainsSet, double hmmInitConstant) {
    int i, j;
    int statesCount = 0, symbolsCount = 0;

    /** Computing PI the numbers of states and symbols*/
    for (i = 0; i < chainsSet.count; ++i) {
        for (j = 0; j < chainsSet.chain[i].length; ++j) {
            if (chainsSet.chain[i].state[j] > statesCount) statesCount = chainsSet.chain[i].state[j] + 1; // +1 because states are 0-indexed
            if (chainsSet.chain[i].observation[j] > symbolsCount) symbolsCount = chainsSet.chain[i].observation[j] + 1;
        }
    }
    Hmm model;
    model.statesCount = statesCount;
    model.symbolsCount = symbolsCount;

    double start_i, transit_i, transit_ij, observation_i, observation_ij, lineSum, remainder;

    /** Computing PI */
    lineSum = 0;
    for (i = 0; i < model.statesCount; i++) {
        start_i = 0;
        for (j = 0; j < chainsSet.count; j++) {
            if (chainsSet.chain[j].state[0] == i)start_i++;
        }
        model.PI[i] = start_i / (chainsSet.count + hmmInitConstant);
        lineSum += model.PI[i];
    }
    remainder = (1 - lineSum) / model.statesCount;
    for (i = 0; i < model.statesCount; i++) {
        model.PI[i] += remainder;
    }

    /** Computing A */
    for (i = 0; i < model.statesCount; i++) {
        lineSum = 0;
        for (j = 0; j < model.statesCount; j++) {
            transit_i = transit_ij = 0;
            for (i = 0; i < chainsSet.count; i++) {
                for (j = 0; j < chainsSet.chain[i].length - 1; j++) {
                    if (chainsSet.chain[i].state[j] == i) {
                        transit_i++;
                        if (chainsSet.chain[i].state[j + 1] == j)transit_ij++;
                    }
                }
            }
            model.A[i][j] = transit_ij / (transit_i + hmmInitConstant);
            lineSum += model.A[i][j];
        }
        remainder = (1 - lineSum) / model.statesCount;
        for (j = 0; j < model.statesCount; j++) {
            model.A[i][j] += remainder;
        }
    }

    /** Computing B */
    for (i = 0; i < model.statesCount; i++) {
        lineSum = 0;
        for (j = 0; j < model.symbolsCount; j++) {
            observation_i = observation_ij = 0;
            for (i = 0; i < chainsSet.count; i++) {
                for (j = 0; j < chainsSet.chain[i].length; j++) {
                    if (chainsSet.chain[i].state[j] == i) {
                        observation_i++;
                        if (chainsSet.chain[i].observation[j] == j)observation_ij++;
                    }
                }
            }
            model.B[i][j] = observation_ij / (observation_i + hmmInitConstant);
            lineSum += model.B[i][j];
        }
        remainder = (1 - lineSum) / model.symbolsCount;
        for (j = 0; j < model.symbolsCount; j++) {
            model.B[i][j] += remainder;
        }
    }
    return model;
}

Hmm hmmClone(Hmm orig) {
    Hmm result;
    result.statesCount = orig.statesCount;
    result.symbolsCount = orig.symbolsCount;

    int i;
    for (i = 0; i < orig.statesCount; i++) {
        memcpy(result.A[i], orig.A[i], orig.statesCount * sizeof(orig.A[i][0]));
        memcpy(result.B[i], orig.B[i], orig.symbolsCount * sizeof(orig.B[i][0]));
    }
    memcpy(result.PI, orig.PI, orig.statesCount * sizeof(orig.PI[0]));
    memcpy(result.PHI, orig.PHI, orig.statesCount * sizeof(orig.PHI[0]));
    return result;
}

void hmmToBuffer(Hmm model, double buffer[]) {
    int i;
    double *runner = buffer;
    runner[0] = (double) model.statesCount;
    runner[1] = (double) model.symbolsCount;
    runner += 2;
    memcpy(runner, model.PI, model.statesCount * sizeof(model.PI[0]));
    runner += model.statesCount;
    for (i = 0; i < model.statesCount; i++) {
        memcpy(runner, model.A[i], model.statesCount * sizeof(model.A[i][0]));
        runner += model.statesCount;
    }
    for (i = 0; i < model.statesCount; i++) {
        memcpy(runner, model.B[i], model.symbolsCount * sizeof(model.B[i][0]));
        runner += model.symbolsCount;
    }
    memcpy(runner, model.PHI, model.statesCount * sizeof(model.PHI[0]));
}

void hmmFromBuffer(Hmm model, double buffer[]) {
    int i;
    model.statesCount = (int) buffer[0];
    model.symbolsCount = (int) buffer[1];
    double *runner = buffer + 2;
    memcpy(model.PI, runner, model.statesCount * sizeof(runner[0]));
    runner += model.statesCount;
    for (i = 0; i < model.statesCount; i++) {
        memcpy(model.A[i], runner, model.statesCount * sizeof(runner[0]));
        runner += model.statesCount;
    }
    for (i = 0; i < model.statesCount; i++) {
        memcpy(model.B[i], runner, model.symbolsCount * sizeof(runner[0]));
        runner += model.symbolsCount;
    }
    memcpy(model.PHI, runner, model.statesCount * sizeof(runner[0]));
}

void computeStationaryDistribution(int size, double *matrix[], double phi[]) {
    int i, j, k, count = 0;
    double temp[size][size], result[size][size];
    for (i = 0; i < size; ++i) {
        memcpy(result[i], matrix[i], size * sizeof(double));
    }
    while (count < size) {
        for (i = 0; i < size; ++i) {
            memcpy(temp[i], result[i], size * sizeof(double));
        }

        for (i = 0; i < size; i++) {
            for (j = 0; j < size; j++) {
                result[i][j] = 0;
                for (k = 0; k < size; k++) {
                    result[i][j] += (temp[i][k] * temp[k][j]);
                }
            }
        }

        // check if phi is equal to the first row of the last result
        count = 0;
        for (i = 0; i < size; ++i) {
            phi[i] = 0.0;
            for (j = 0; j < size; ++j) {
                phi[i] += result[0][j] * matrix[j][i];
            }
            if (phi[i] == result[0][j])count++;
        }
    }
}

void sortAscendant(int length, double array[]) {
    double min, temp;
    int minIndex, i, j;
    for (i = 0; i < length - 1; i++) {
        min = array[i];
        minIndex = i;
        for (j = i + 1; j < length; j++) {
            if (min - array[j] > 0.0) {
                minIndex = j;
                min = array[j];
            }
        }
        temp = array[i];
        array[i] = array[minIndex];
        array[minIndex] = temp;
    }
}

double giniIndex(int length, double array[]) {
    int i;
    double norm = 0.0, temp = 0.0, result;
    sortAscendant(length, array);
    for (i = 0; i < length; i++) {
        norm += array[i];
    }
    if (norm == 0.0)return 0.0;
    for (i = 1; i <= length; i++) {
        temp += (array[i - 1] / norm) * ((length - i + 0.5) / (length - 1.0));
    }
    result = (length / (length - 1.0)) - 2.0 * temp;
    return result;
}

double hmmSahraeianSimilarity(Hmm model1, Hmm model2) {
    int i, j, k, QRows = model1.statesCount, QColumns = model2.statesCount;
    double Q[QRows][QColumns], row[QRows], column[QColumns];
    double denomQ = 0., rowsGiniIndexSum = 0., columnsGiniIndexSum = 0.;
    for (i = 0; i < QRows; i++) {
        for (j = 0; j < QColumns; j++) {
            Q[i][j] = 0;
            for (k = 0; k < model1.symbolsCount; k++) {
                if (k < model2.symbolsCount)
                    Q[i][j] += sqrt(model1.B[i][k] * model2.B[j][k]);
            }
            Q[i][j] = -2 * log(Q[i][j] + DBL_MIN);
            Q[i][j] = model1.PHI[i] * model2.PHI[j] * exp(-2 * Q[i][j]);
            denomQ += Q[i][j];
        }
    }

    for (i = 0; i < QRows; i++) {
        for (j = 0; j < QColumns; j++) {
            Q[i][j] /= denomQ;
        }
    }

    for (i = 0; i < QRows; i++) {
        for (j = 0; j < QColumns; j++) {
            row[j] = Q[i][j];
        }
        rowsGiniIndexSum += giniIndex(QRows, row);
    }
    for (j = 0; j < QColumns; j++) {
        for (i = 0; i < QRows; i++) {
            column[i] = Q[i][j];
        }
        columnsGiniIndexSum += giniIndex(QColumns, column);
    }

    return 0.5 * (rowsGiniIndexSum / QRows + columnsGiniIndexSum / QColumns) * 100;
}

double Forward(Hmm model, Sequence observation, long double alpha[][model.statesCount]) {
    int i, j, t;
    long double proba = 0.0;
    for (i = 0; i < model.statesCount; i++) {
        alpha[0][i] = model.PI[i] * model.B[i][observation.data[0]];
    }
    for (t = 0; t < observation.length - 1; t++) {
        for (j = 0; j < model.statesCount; j++) {
            alpha[t + 1][j] = 0;
            for (i = 0; i < model.statesCount; i++) {
                alpha[t + 1][j] += alpha[t][i] * model.A[i][j];
            }
            alpha[t + 1][j] *= model.B[j][observation.data[t + 1]];
        }
    }
    for (i = 0; i < model.statesCount; i++) {
        proba += alpha[observation.length - 1][i];
    }
    return (double) proba;
}

void Backward(Hmm model, Sequence observation, long double beta[][model.statesCount]) {
    int i, j, t;
    for (i = 0; i < model.statesCount; i++) {
        beta[observation.length - 1][i] = 1;
    }
    for (t = observation.length - 1; t > 0; t--) {
        for (i = 0; i < model.statesCount; i++) {
            beta[t - 1][i] = 0;
            for (j = 0; j < model.statesCount; j++) {
                beta[t - 1][i] += beta[t][j] * model.A[i][j] * model.B[j][observation.data[t]];
            }
        }
    }

}

void Gamma(Hmm model, Sequence observation, long double alpha[][model.statesCount], long double beta[][model.statesCount],
           long double gamma[][model.statesCount], double proba) {

    int t, j;
    for (t = 0; t < observation.length; t++) {
        for (j = 0; j < model.statesCount; j++) {
            gamma[t][j] = (alpha[t][j] * beta[t][j]) / (proba + DBL_MIN);
        }
    }
}

void
Xi(Hmm model, Sequence observation, long double alpha[][model.statesCount], long double beta[][model.statesCount],
   long double xi[][model.statesCount][model.statesCount], double proba) {

    int i, j, t;
    for (t = 0; t < observation.length - 1; t++) {
        for (i = 0; i < model.statesCount; i++) {
            for (j = 0; j < model.statesCount; j++) {
                xi[t][i][j] = (alpha[t][i] * model.A[i][j] * model.B[j][observation.data[t + 1]] * beta[t + 1][j]);
            }
            xi[t][i][j] /= (proba + DBL_MIN);
        }
    }
}

Hmm standardBaumWelch(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold) {
    int i, k, j, t;
    int iteration = 0;
    double proba, deltaProba, sequenceProba[observations.count];
    int maxSeqLength = getMaxSequenceLength(observations);

    long double alpha[observations.count][maxSeqLength][model.statesCount];
    long double beta[observations.count][maxSeqLength][model.statesCount];
    long double gamma[observations.count][maxSeqLength][model.statesCount];
    long double xi[observations.count][maxSeqLength][model.statesCount][model.statesCount];

    long double numPI[model.statesCount], denA[model.statesCount], denB[model.statesCount],
            numA[model.statesCount][model.statesCount], numB[model.statesCount][model.symbolsCount];

    /** The model is initially considered trained */
    Hmm trainedModel = hmmClone(model);

    /** Evaluating the global and partial observation probabilities according to the initial model */
    proba = 1.0;
    for (k = 0; k < observations.count; ++k) {
        sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
        proba *= sequenceProba[k];
    }

    do {
        /** Computing model re-estimation variables */
        memset(denA, 0, sizeof(denA));
        memset(denB, 0, sizeof(denB));
        memset(numA, 0, sizeof(numA));
        memset(numB, 0, sizeof(numB));
        memset(numPI, 0, sizeof(numPI));

        for (k = 0; k < observations.count; k++) {
            Backward(trainedModel, observations.sequence[k], beta[k]);
            Gamma(trainedModel, observations.sequence[k], alpha[k], beta[k], gamma[k], sequenceProba[k]);
            Xi(trainedModel, observations.sequence[k], alpha[k], beta[k], xi[k], sequenceProba[k]);
            for (i = 0; i < trainedModel.statesCount; i++) {
                numPI[i] += gamma[k][0][i];
                for (j = 0; j < trainedModel.statesCount; ++j) {
                    for (t = 0; t < observations.sequence[k].length - 1; ++t) {
                        numA[i][j] += xi[k][t][i][j];
                        denA[i] += gamma[k][t][i];
                        denB[i] += gamma[k][t][i];
                    }
                    denB[i] += gamma[k][t + 1][i];
                }
                for (j = 0; j < trainedModel.symbolsCount; ++j) {
                    for (t = 0; t < observations.sequence[k].length; ++t) {
                        if (observations.sequence[k].data[t] == j)
                            numB[i][j] += gamma[k][t][i];
                    }
                }
            }
        }

        /** Updating model parameters */
        for (i = 0; i < model.statesCount; ++i) {
            trainedModel.PI[i] = (double) numPI[i] / observations.count;
            for (j = 0; j < model.statesCount; ++j) {
                /** LDBL_MIN is added to the denominator to avoid potential division by 0. */
                trainedModel.A[i][j] = (double) (numA[i][j] / (denA[i] + LDBL_MIN));
            }
            for (j = 0; j < model.symbolsCount; ++j) {
                trainedModel.B[i][j] = (double) (numB[i][j] / (denB[i] + LDBL_MIN));
            }
        }

        /** Evaluating the global and partial observation probabilities according to the newly updated model */
        double temp = 1.0;
        for (k = 0; k < observations.count; ++k) {
            sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
            temp *= sequenceProba[k];
        }

        /** Evaluating the probability variation. Since the updated model is the input for the next potential iteration
         *  its the probability of observing the sequences according to it will
         * */
        deltaProba = temp - proba;
        proba = temp;
        iteration++;
    } while (iteration < maxIterations && deltaProba > probaThreshold);

    return trainedModel;
}

Hmm percentageBaumWelch(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold, double percentageThreshold) {
    int i, k, j, t;
    int iteration = 0;
    double percentage, sequenceProba[observations.count];
    int maxSeqLength = getMaxSequenceLength(observations);

    long double alpha[observations.count][maxSeqLength][model.statesCount];
    long double beta[observations.count][maxSeqLength][model.statesCount];
    long double gamma[observations.count][maxSeqLength][model.statesCount];
    long double xi[observations.count][maxSeqLength][model.statesCount][model.statesCount];

    long double numPI[model.statesCount], denA[model.statesCount], denB[model.statesCount],
            numA[model.statesCount][model.statesCount], numB[model.statesCount][model.symbolsCount];

    /** The model is initially considered trained */
    Hmm trainedModel = hmmClone(model);

    /** Evaluating the global and partial observation probabilities according to the initial model */
    for (k = 0; k < observations.count; ++k) {
        sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
    }

    do {
        /** Computing model re-estimation variables */
        memset(denA, 0, sizeof(denA));
        memset(denB, 0, sizeof(denB));
        memset(numA, 0, sizeof(numA));
        memset(numB, 0, sizeof(numB));
        memset(numPI, 0, sizeof(numPI));
        for (k = 0; k < observations.count; k++) {
            Backward(trainedModel, observations.sequence[k], beta[k]);
            Gamma(trainedModel, observations.sequence[k], alpha[k], beta[k], gamma[k], sequenceProba[k]);
            Xi(trainedModel, observations.sequence[k], alpha[k], beta[k], xi[k], sequenceProba[k]);
            for (i = 0; i < trainedModel.statesCount; i++) {
                numPI[i] += gamma[k][0][i];
                for (j = 0; j < trainedModel.statesCount; ++j) {
                    for (t = 0; t < observations.sequence[k].length - 1; ++t) {
                        numA[i][j] += xi[k][t][i][j];
                        denA[i] += gamma[k][t][i];
                        denB[i] += gamma[k][t][i];
                    }
                    denB[i] += gamma[k][t + 1][i];
                }
                for (j = 0; j < trainedModel.symbolsCount; ++j) {
                    for (t = 0; t < observations.sequence[k].length; ++t) {
                        if (observations.sequence[k].data[t] == j)
                            numB[i][j] += gamma[k][t][i];
                    }
                }
            }
        }

        /** Updating model parameters */
        for (i = 0; i < model.statesCount; ++i) {
            trainedModel.PI[i] = (double) numPI[i] / observations.count;
            for (j = 0; j < model.statesCount; ++j) {
                /** LDBL_MIN is added to the denominator to avoid potential division by 0. */
                trainedModel.A[i][j] = (double) (numA[i][j] / (denA[i] + LDBL_MIN));
            }
            for (j = 0; j < model.symbolsCount; ++j) {
                trainedModel.B[i][j] = (double) (numB[i][j] / (denB[i] + LDBL_MIN));
            }
        }

        /** Evaluating the percentage of observation probabilities meeting the threshold condition according to the newly updated model */
        int okProbaCount = 0;
        double temp, deltaProba;
        for (k = 0; k < observations.count; ++k) {
            temp = Forward(trainedModel, observations.sequence[k], alpha[k]);
            deltaProba = temp - sequenceProba[k];
            sequenceProba[k] = temp;
            if (deltaProba > 0 && deltaProba < probaThreshold)okProbaCount++;
        }
        percentage = okProbaCount * 100.0 / observations.count;
        iteration++;
    } while (iteration < maxIterations && percentage < percentageThreshold);

    return trainedModel;
}


