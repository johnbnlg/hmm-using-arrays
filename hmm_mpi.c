#include "hmm_mpi.h"

void mpiSendHmm(Hmm model, int dest, int tag, MPI_Comm com) {
    int bufferSize = (2 + model.statesCount + model.symbolsCount) * model.statesCount + 2;
    double buffer[bufferSize];
    hmmToBuffer(model, buffer);
    MPI_Send(&bufferSize, 1, MPI_INT, dest, tag, com);
    MPI_Send(buffer, bufferSize, MPI_DOUBLE, dest, tag, com);
}

Hmm mpiReceiveHmm(int source, int tag, MPI_Comm com) {
    int bufferSize;
    MPI_Status status;
    MPI_Recv(&bufferSize, 1, MPI_INT, source, tag, com, &status);
    double buffer[bufferSize];
    MPI_Recv(buffer, bufferSize, MPI_DOUBLE, source, tag, com, &status);
    Hmm model = newHmm((int) buffer[0], (int) buffer[1]);
    hmmFromBuffer(model, buffer);
    return model;
}

void mpiSendSequencesSet(SequencesSet set, int dest, int tag, MPI_Comm com) {
    int bufferSize = sequencesSetBufferSize(set);
    int buffer[bufferSize];
    sequencesSetToBuffer(set, buffer);
    MPI_Send(&bufferSize, 1, MPI_INT, dest, tag, com);
    MPI_Send(buffer, bufferSize, MPI_INT, dest, tag, com);
}

SequencesSet mpiReceiveSequencesSet(int source, int tag, MPI_Comm com) {
    int bufferSize;
    MPI_Status status;
    MPI_Recv(&bufferSize, 1, MPI_INT, source, tag, com, &status);
    int buffer[bufferSize];
    MPI_Recv(buffer, bufferSize, MPI_INT, source, tag, com, &status);
    return sequencesSetFromBuffer(buffer);
}

Hmm
mpiStandardBaumWelchMaster(Hmm model, SequencesSet allObservations, int maxIterations, double probaThreshold, MPI_Comm com) {
    int i, k, j, t, p, procCount, goToNextIteration;
    int iteration = 0;
    MPI_Status status;

    MPI_Comm_size(com, &procCount);
    SequencesSet subsets[procCount];
    sequencesSetSplit(allObservations, procCount, subsets);

    /** Dispatching initial model and sequences set */
    for (p = 1; p < procCount; ++p) {
        mpiSendHmm(model, p, 0, com);
        mpiSendSequencesSet(subsets[p], p, 0, com);
    }

    SequencesSet observations = subsets[0];
    int bufferSize = model.statesCount * (3 + model.statesCount + model.symbolsCount);
    long double buffer[bufferSize];
    double proba, deltaProba, sequenceProba[observations.count];
    long double **alpha[observations.count], **beta[observations.count], **gamma[observations.count], ***xi[observations.count];
    long double numPI[model.statesCount], denA[model.statesCount], denB[model.statesCount],
            numA[model.statesCount][model.statesCount], numB[model.statesCount][model.symbolsCount];

    /** Allocating memory for alpha, beta, gamma and xi */
    mallocAlphaBetaGammaXi(model, observations, alpha, beta, gamma, xi);

    /** The model is initially considered trained */
    Hmm trainedModel = hmmClone(model);

    /** Evaluating the global and partial observation probabilities according to the initial model */
    proba = 1.0;
    for (k = 0; k < observations.count; ++k) {
        sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
        proba *= sequenceProba[k];
    }

    /** Taking in consideration slaves' contribution for the global probability */
    for (p = 1; p < procCount; ++p) {
        double temp;
        MPI_Recv(&temp, 1, MPI_DOUBLE, p, 0, com, &status);
        proba *= temp;
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
        /** Receiving slaves' contribution for model re-estimation */
        long double *runner = buffer;
        for (p = 1; p < procCount; ++p) {
            MPI_Recv(buffer, bufferSize, MPI_LONG_DOUBLE, p, 0, com, &status);
            for (i = 0; i < trainedModel.statesCount; ++i) {
                numPI[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                denA[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                denB[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                for (j = 0; j < trainedModel.statesCount; ++j) {
                    numA[i][j] += *runner;
                    runner++;
                }
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                for (j = 0; j < trainedModel.symbolsCount; ++j) {
                    numB[i][j] += *runner;
                    runner++;
                }
            }
        }

        /** Updating model parameters */
        for (i = 0; i < trainedModel.statesCount; ++i) {
            trainedModel.PI[i] = (double) numPI[i] / observations.count;
            for (j = 0; j < trainedModel.statesCount; ++j) {
                /** LDBL_MIN is added to the denominator to avoid potential division by 0. */
                trainedModel.A[i][j] = (double) (numA[i][j] / (denA[i] + LDBL_MIN));
            }
            for (j = 0; j < trainedModel.symbolsCount; ++j) {
                trainedModel.B[i][j] = (double) (numB[i][j] / (denB[i] + LDBL_MIN));
            }
        }

        /** Dispatching the newly updated model */
        for (p = 1; p < procCount; ++p) {
            mpiSendHmm(trainedModel, p, 0, com);
        }

        /** Evaluating the global and partial observation probabilities according to the newly updated model */
        double newProba = 1.0;
        for (k = 0; k < observations.count; ++k) {
            sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
            newProba *= sequenceProba[k];
        }

        /** Taking in consideration slaves' contribution for the global probability */
        for (p = 1; p < procCount; ++p) {
            double temp;
            MPI_Recv(&temp, 1, MPI_DOUBLE, p, 0, com, &status);
            newProba *= temp;
        }

        /** Evaluating the probability variation. Since the updated model is the input for the next potential iteration
         *  its the probability of observing the sequences according to it will
         * */
        deltaProba = newProba - proba;
        proba = newProba;
        iteration++;
        goToNextIteration = (iteration < maxIterations && deltaProba > probaThreshold);
        for (p = 1; p < procCount; ++p) {
            MPI_Send(&goToNextIteration, 1, MPI_INT, p, 0, com);
        }
    } while (goToNextIteration);

    /** Deallocating the memory previously allocated to alpha, beta, gamma and xi */
    freeAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);
    return trainedModel;
}

void mpiStandardBaumWelchSlave(MPI_Comm com) {
    int i, k, j, t, goToNextIteration;
    MPI_Status status;

    /** Receiving initial model and sequences set */
    Hmm trainedModel = mpiReceiveHmm(0, 0, com);
    SequencesSet observations = mpiReceiveSequencesSet(0, 0, com);

    int bufferSize = trainedModel.statesCount * (3 + trainedModel.statesCount + trainedModel.symbolsCount);
    long double buffer[bufferSize];
    double proba, sequenceProba[observations.count];
    long double **alpha[observations.count], **beta[observations.count], **gamma[observations.count], ***xi[observations.count];
    long double numPI[trainedModel.statesCount], denA[trainedModel.statesCount], denB[trainedModel.statesCount],
            numA[trainedModel.statesCount][trainedModel.statesCount], numB[trainedModel.statesCount][trainedModel.symbolsCount];

    /** Allocating memory for alpha, beta, gamma and xi */
    mallocAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);

    /** Evaluating the global and partial observation probabilities according to the initial model */
    proba = 1.0;
    for (k = 0; k < observations.count; ++k) {
        sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
        proba *= sequenceProba[k];
    }
    MPI_Send(&proba, 1, MPI_DOUBLE, 0, 0, com);

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
        /** Sending this slave's contribution to the master for model re-estimation */
        long double *runner = buffer;
        memcpy(runner, numPI, sizeof(numPI));
        runner += trainedModel.statesCount;
        memcpy(runner, denA, sizeof(denA));
        runner += trainedModel.statesCount;
        memcpy(runner, denB, sizeof(denB));
        runner += trainedModel.statesCount;
        for (i = 0; i < trainedModel.statesCount; ++i) {
            memcpy(runner, numA[i], sizeof(numA[i]));
            runner += trainedModel.statesCount;
        }
        for (i = 0; i < trainedModel.statesCount; ++i) {
            memcpy(runner, numB[i], sizeof(numB[i]));
            runner += trainedModel.symbolsCount;
        }
        MPI_Send(buffer, bufferSize, MPI_LONG_DOUBLE, 0, 0, com);

        /** Receiving the newly updated model */
        freeHmm(trainedModel);
        trainedModel = mpiReceiveHmm(0, 0, com);

        proba = 1.0;
        for (k = 0; k < observations.count; ++k) {
            sequenceProba[k] = Forward(trainedModel, observations.sequence[k], alpha[k]);
            proba *= sequenceProba[k];
        }

        /** Sending the partial probability computed by this slave*/
        MPI_Send(&proba, 1, MPI_DOUBLE, 0, 0, com);

        /** Receiving the signal whether to go or not to the next iteration from the master */
        MPI_Recv(&goToNextIteration, 1, MPI_INT, 0, 0, com, &status);
    } while (goToNextIteration);

    /** Deallocating the memory previously allocated to alpha, beta, gamma and xi */
    freeAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);
}

Hmm mpiPercentageBaumWelchMaster(Hmm model, SequencesSet allObservations, int maxIterations, double probaThreshold, double percentageThreshold, MPI_Comm com) {
    int i, k, j, t, p, procCount, goToNextIteration;
    int iteration = 0;
    MPI_Status status;

    MPI_Comm_size(com, &procCount);
    SequencesSet subsets[procCount];
    sequencesSetSplit(allObservations, procCount, subsets);

    /** Dispatching initial model, probability threshold and sequences set */
    for (p = 1; p < procCount; ++p) {
        MPI_Send(&probaThreshold, 1, MPI_DOUBLE, p, 0, com);
        mpiSendHmm(model, p, 0, com);
        mpiSendSequencesSet(subsets[p], p, 0, com);
    }

    SequencesSet observations = subsets[0];
    int bufferSize = model.statesCount * (3 + model.statesCount + model.symbolsCount);
    long double buffer[bufferSize];
    double percentage, sequenceProba[observations.count];
    long double **alpha[observations.count], **beta[observations.count], **gamma[observations.count], ***xi[observations.count];
    long double numPI[model.statesCount], denA[model.statesCount], denB[model.statesCount],
            numA[model.statesCount][model.statesCount], numB[model.statesCount][model.symbolsCount];

    /** Allocating memory for alpha, beta, gamma and xi */
    mallocAlphaBetaGammaXi(model, observations, alpha, beta, gamma, xi);

    /** The model is initially considered trained */
    Hmm trainedModel = hmmClone(model);

    /** Evaluating observation probabilities according to the initial model */
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

        /** Receiving slaves' contributions for model re-estimation */
        long double *runner = buffer;
        for (p = 1; p < procCount; ++p) {
            MPI_Recv(buffer, bufferSize, MPI_LONG_DOUBLE, p, 0, com, &status);
            for (i = 0; i < trainedModel.statesCount; ++i) {
                numPI[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                denA[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                denB[i] += *runner;
                runner++;
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                for (j = 0; j < trainedModel.statesCount; ++j) {
                    numA[i][j] += *runner;
                    runner++;
                }
            }
            for (i = 0; i < trainedModel.statesCount; ++i) {
                for (j = 0; j < trainedModel.symbolsCount; ++j) {
                    numB[i][j] += *runner;
                    runner++;
                }
            }
        }

        /** Updating model parameters */
        for (i = 0; i < trainedModel.statesCount; ++i) {
            trainedModel.PI[i] = (double) numPI[i] / observations.count;
            for (j = 0; j < trainedModel.statesCount; ++j) {
                /** LDBL_MIN is added to the denominator to avoid potential division by 0. */
                trainedModel.A[i][j] = (double) (numA[i][j] / (denA[i] + LDBL_MIN));
            }
            for (j = 0; j < trainedModel.symbolsCount; ++j) {
                trainedModel.B[i][j] = (double) (numB[i][j] / (denB[i] + LDBL_MIN));
            }
        }

        /** Dispatching the newly updated model */
        for (p = 1; p < procCount; ++p) {
            mpiSendHmm(trainedModel, p, 0, com);
        }

        /** Evaluating observation probabilities and the number of sequences satisfying the probability threshold according to the newly updated model */
        int okProbaCount = 0;
        for (k = 0; k < observations.count; ++k) {
            double temp, deltaProba;
            temp = Forward(trainedModel, observations.sequence[k], alpha[k]);
            deltaProba = temp - sequenceProba[k];
            sequenceProba[k] = temp;
            if (deltaProba > 0 && deltaProba < probaThreshold)okProbaCount++;
        }
        /** Receiving the number of sequences satisfying the probability threshold from slave and add to the master's one*/
        for (p = 1; p < procCount; ++p) {
            int temp;
            MPI_Recv(&temp, 1, MPI_INT, p, 0, com, &status);
            okProbaCount += temp;
        }
        percentage = okProbaCount * 100.0 / observations.count;
        iteration++;

        /** Sending the signal whether to go or not to the next iteration to slaves */
        goToNextIteration = (iteration < maxIterations && percentage < percentageThreshold);
        for (p = 1; p < procCount; ++p) {
            MPI_Send(&goToNextIteration, 1, MPI_INT, p, 0, com);
        }
    } while (goToNextIteration);

    /** Deallocating the memory previously allocated to alpha, beta, gamma and xi */
    freeAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);
    return trainedModel;
}

void mpiPercentageBaumWelchSlave(MPI_Comm com) {
    int i, k, j, t, goToNextIteration;
    MPI_Status status;
    double probaThreshold;
    /** Receiving initial model, probability threshold and sequences set */
    MPI_Recv(&probaThreshold, 1, MPI_DOUBLE, 0, 0, com, &status);
    Hmm trainedModel = mpiReceiveHmm(0, 0, com);
    SequencesSet observations = mpiReceiveSequencesSet(0, 0, com);

    int bufferSize = trainedModel.statesCount * (3 + trainedModel.statesCount + trainedModel.symbolsCount);
    long double buffer[bufferSize];
    double proba, sequenceProba[observations.count];
    long double **alpha[observations.count], **beta[observations.count], **gamma[observations.count], ***xi[observations.count];
    long double numPI[trainedModel.statesCount], denA[trainedModel.statesCount], denB[trainedModel.statesCount],
            numA[trainedModel.statesCount][trainedModel.statesCount], numB[trainedModel.statesCount][trainedModel.symbolsCount];

    /** Allocating memory for alpha, beta, gamma and xi */
    mallocAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);

    /** Evaluating observation probabilities according to the initial model */
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

        /** Sending slave's' contributions to the master for model re-estimation */
        long double *runner = buffer;
        memcpy(runner, numPI, sizeof(numPI));
        runner += trainedModel.statesCount;
        memcpy(runner, denA, sizeof(denA));
        runner += trainedModel.statesCount;
        memcpy(runner, denB, sizeof(denB));
        runner += trainedModel.statesCount;
        for (i = 0; i < trainedModel.statesCount; ++i) {
            memcpy(runner, numA[i], sizeof(numA[i]));
            runner += trainedModel.statesCount;
        }
        for (i = 0; i < trainedModel.statesCount; ++i) {
            memcpy(runner, numB[i], sizeof(numB[i]));
            runner += trainedModel.symbolsCount;
        }
        MPI_Send(buffer, bufferSize, MPI_LONG_DOUBLE, 0, 0, com);

        /** Receiving the newly updated model from the master*/
        freeHmm(trainedModel);
        trainedModel = mpiReceiveHmm(0, 0, com);

        /** Evaluating observation probabilities and the number of sequences satisfying the probability threshold according to the newly updated model */
        int okProbaCount = 0;
        double temp, deltaProba;
        for (k = 0; k < observations.count; ++k) {
            temp = Forward(trainedModel, observations.sequence[k], alpha[k]);
            deltaProba = temp - sequenceProba[k];
            sequenceProba[k] = temp;
            if (deltaProba > 0 && deltaProba < probaThreshold)okProbaCount++;
        }

        /** Sending the number of sequences satisfying the probability threshold to the master*/
        MPI_Send(&okProbaCount, 1, MPI_INT, 0, 0, com);

        /** Receiving the signal whether to go or not to the next iteration from the master */
        MPI_Recv(&goToNextIteration, 1, MPI_INT, 0, 0, com, &status);
    } while (goToNextIteration);

    /** Deallocating the memory previously allocated to alpha, beta, gamma and xi */
    freeAlphaBetaGammaXi(trainedModel, observations, alpha, beta, gamma, xi);
}



