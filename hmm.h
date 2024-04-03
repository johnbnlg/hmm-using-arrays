#ifndef HMM_H
#define HMM_H

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

#define MAX_STATES_COUNT 100 /** Maximum states count */
#define MAX_SYMBOLS_COUNT 100 /** Maximum symbols count */
#define MAX_SEQUENCES_COUNT 100 /** Maximum sequences (of states or symbols) count */
#define MAX_SEQUENCE_LENGTH 100 /** Maximum sequence (of states or symbols) length */

/**
  * Object model corresponding to a Hidden Markov Model (Hmm)
  */
typedef struct Hmm {
    int statesCount;
    int symbolsCount;
    double A[MAX_STATES_COUNT][MAX_STATES_COUNT]; /** Matrix of probabilities of transition between states (of format N x N) */
    double B[MAX_STATES_COUNT][MAX_SYMBOLS_COUNT]; /** Matrix of probabilities of observation of symbols knowing the state (of format N x M) */
    double PI[MAX_STATES_COUNT]; /** Vector of initial state probabilities (of size N) */
    double PHI[MAX_STATES_COUNT]; /** Stationary distribution vector (of size N) */
} Hmm;

/**
  * Object model corresponding to a sequence
  */
typedef struct Sequence {
    int length;
    int data[MAX_SEQUENCE_LENGTH];
} Sequence;

/**
  * Object model corresponding to a set of sequences
  */
typedef struct SequencesSet {
    int count; /** Sequences count */
    Sequence sequence[MAX_SEQUENCES_COUNT]; /** Sequences */
} SequencesSet;

/**
  * Object model corresponding to a Markov chain
  */
typedef struct MarkovChain {
    int length; /** Markov chain length */
    int state[MAX_SEQUENCE_LENGTH]; /** Chain of states */
    int observation[MAX_SEQUENCE_LENGTH]; /** Observations associated respectively with the different states */
} MarkovChain;

/**
 * Object model corresponding to a set of Markov chains
 */
typedef struct MarkovChainsSet {
    int count; /** Chains count */
    MarkovChain chain[MAX_SEQUENCES_COUNT]; /** Markov chains */
} MarkovChainsSet;

/**
 * Compute the buffer size to store a sequences set.
 * @param set : The whose buffer size is computed
 * @return : The buffer size of the given set
 */
int sequencesSetBufferSize(SequencesSet set);

/**
 * Serialize a sequences set into a buffer. The first element of the buffer is
 * the number of sequences. It's followed by the sequences lengths and
 * the sequences data are at the end in the same order as their lengths.
 * @param set : The set to be serialized
 * @param buffer : The buffer where to store the sequences set
 */
void sequencesSetToBuffer(SequencesSet set, int buffer[]);

/**
 * Extract a sequences set previously stored in a buffer
 * @param buffer : The buffer containing the sequences set.
 * @return : The extracted sequences set.
 */
SequencesSet sequencesSetFromBuffer(int buffer[]);


/**
 * Split a large sequences set into smaller ones
 * @param set : The given sequences set
 * @param subsetsCount : The expected number of subsets.
 * @param subsets : An array containing the resulting subsets.
 */
void sequencesSetSplit(SequencesSet set, int subsetsCount, SequencesSet subsets[]);

/**
  * Creation and initialization of an Hmm from a set of Markov chains
  * @param N: Number of model states
  * @param M: Number of pattern symbols
  * @param chainsSet: set of Markov chains
  * @return: model created
  */
Hmm hmmFromMarkovChainSet(MarkovChainsSet chainsSet, double hmmInitConstant);

/**
  * Cloning a model
  * @param orig: destination model
  * @return: model clone passed as source model parameter.
  */
Hmm hmmClone(Hmm orig);

/**
  * Hmm serialization
  * In the buffer, model parameters organized in this order: N, M, PI, A, B, PHI
  * M and N are cast to double on store.
  * @param model: Model to compact
  * @param buffer: Store/transmit buffer
  */
void hmmToBuffer(Hmm model, double buffer[]);

/**
  * Hmm deserialization
   * @param model: Obtained model storage
  * @param buffer: A buffer containing the model parameters organized in this order:  states count, symbols count, PI, A, B, PHI
  * states and symbols counts are casted to double on store.
  */
void hmmFromBuffer(Hmm model, double buffer[]);


/**
  * Update of the stationary distribution of a given model.
  * @param model: Given model.
  */
void computeStationaryDistribution(int size, double *matrix[], double phi[]);

/**
 * Sorts an array of doubles in ascending order
 * @param length: The length of the array
 * @param array: The array to be sorted
 */
void sortAscendant(int length, double array[]);

/**
 * Compute the Gini index for an array of doubles
 * @param array: Array considered
 * @param length: size of the Array considered
 * @return: the calculated value of the Gini index
 */
double giniIndex(int length, double array[]);

/**
  * Computate of the similarity rate between two models according to SME Sahraeian(2010)'s algorithm
  * @param model1: First model.
  * @param model2: Second model.
  * @return: similarity between the two models.
  */
double hmmSahraeianSimilarity(Hmm model1, Hmm model2);

/**
  * Computation of the alpha matrix (of the Forward Backward procedure) and the
  * probability of observation of a sequence.
  * alpha[t][i] denotes the probability of the partial observation sequence, O[1] • • • O[t], (until time t)and state S[i] at time t, given the model
  * @param model: Given model
  * @param observation: Observations sequence
  * @param alpha: The observation.length * model.statesCount calculated matrix
  * @return : Probability of observing the given sequence according the model
  */
double Forward(Hmm model, Sequence observation, long double *alpha[]);

/**
  * Computation of the beta matrix (of the Forward Backward procedure) for the
  * determination of the probability of observation of a sequence.
  * beta[t][i] denotes the probability of the partial observation sequence, O[t] • • • O[T], (T being the full length of the sequence)
  * and state S[i] at time t, given the model
  * @param model: Given model
  * @param observation: observations sequence
  * @param beta: The observation.length * model.statesCount calculated matrix
  */
void Backward(Hmm model, Sequence observation, long double *beta[]);

/**
  * Calculation of the Gamma parameter to be used for the re-estimation of the model
  * Gamma[t][i] denotes the probability of being in state S[i] at time t, given the observation sequence and the model
  * @param model: Considered model
  * @param observation: observations sequence
  * @param alpha: Forward matrix (observation.length * model.statesCount)
  * @param beta: Backward matrix (observation.length * model.statesCount)
  * @param gamma: The calculated matrix
  * @param proba: Probability of observing the considered sequence knowing the given model
  */
void Gamma(Hmm model, Sequence observation, long double *alpha[], long double *beta[], long double *gamma[], double proba);

/**
  * Calculation of the Xi parameter to be used for the re-estimation of the model
  * Xi[t][i][j] denotes the probability of being in state S[i] at time t, and state S[j] at time t + 1, given the model and the observation sequence
  * @param model: Considered model
  * @param observation: observations sequence
  * @param alpha: Forward matrix (observation.length * model.statesCount)
  * @param beta: Backward matrix (observation.length * model.statesCount)
  * @param xi: The calculated matrix (observation.length * model.statesCount * model.statesCount)
  * @param proba: Probability of observing the considered sequence knowing the given model
  */
void Xi(Hmm model, Sequence observation, long double *alpha[], long double *beta[], long double *xi[observation.length][model.statesCount], double proba);

/**
  * Sequential training of a given model for the recognition of a set of observation sequences
  * using the standard BaumWelch algorithm
  * @param model: Model to train
  * @param observations: set of observation sequences
  * @param maxIterations: training iterations probaThreshold.
  * @param probaThreshold: probability probaThreshold for each sequence.
  * @return: Trained model
  */
Hmm standardBaumWelch(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold);

/**
  * Sequential training of a given model for the recognition of a set of observation sequences
  * using a modify BaumWelch algorithm based on sequences percentage
  * @param model: Model to train
  * @param observations: set of observation sequences
  * @param maxIterations: training iterations probaThreshold.
  * @param probaThreshold: probability threshold for each sequence.
  * @param percentageThreshold: The percentage of sequences that must satisfy the probability threshold condition for the model to be considered trained
  * @return: Trained model
  */
Hmm percentageBaumWelch(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold, double percentageThreshold);

#endif //HMM_H
