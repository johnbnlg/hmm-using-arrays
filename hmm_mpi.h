#ifndef HMM_MPI_H
#define HMM_MPI_H

#include <mpi.h>
#include "hmm.h"


/**
 * Sends a HMM from one processor to another
 * @param model: Model to send
 * @param dest: Number of the destination processor
 * @param tag: Tag identifying the message
 * @param com: Communicator to which both processors belong
 */
void mpiSendHmm(Hmm model, int dest, int tag, MPI_Comm com);

/**
 * Receiving a HMM from another processor
 * @param source: Model source processor number
 * @param tag: Tag identifying the message
 * @param com: Communicator to which both processors belong
 * @return: Model received
 */
Hmm mpiReceiveHmm(int source, int tag, MPI_Comm com);

/**
 * Sending a set of sequences to another processor
 * @param set : sequences to send
 * @param dest : the rank of the receiver.
 * @param tag : The communication tag
 * @param com : The involved communicator
 */
void mpiSendSequencesSet(SequencesSet set, int dest, int tag, MPI_Comm com);

/**
 * Receiving a set of sequences from another processor
 * @param source : The rank of the sender.
 * @param tag : The communication tag
 * @param com : The involved communicator
 * @return
 */
SequencesSet mpiReceiveSequencesSet(int source, int tag, MPI_Comm com);

/**
  * Parallel training(Master part) of a given model for the recognition of a set of observation sequences
  * using the standard BaumWelch algorithm
  * @param model: Model to train
  * @param observations: set of observation sequences
  * @param maxIterations: training iterations probaThreshold.
  * @param probaThreshold: probability probaThreshold for each sequence.
  * @param com: The communicator on which te model is trained
  * @return: Trained model
  */
Hmm mpiStandardBaumWelchMaster(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold, MPI_Comm com);


/**
 * Parallel training(Slave part) of a given model for the recognition of a set of observation sequences
 * using the standard BaumWelch algorithm
 * @param com
 */
void mpiStandardBaumWelchSlave(MPI_Comm com);

/**
  * Parallel training(Master part) of a given model for the recognition of a set of observation sequences
  * using a modify BaumWelch algorithm based on sequences percentage
  * @param model: Model to train
  * @param observations: set of observation sequences
  * @param maxIterations: training iterations probaThreshold.
  * @param probaThreshold: probability threshold for each sequence.
  * @param percentageThreshold: The percentage of sequences that must satisfy the probability threshold condition for the model to be considered trained
  * @param com: The communicator on which te model is trained
  * @return: Trained model
  */
Hmm mpiPercentageBaumWelchMaster(Hmm model, SequencesSet observations, int maxIterations, double probaThreshold, double percentageThreshold, MPI_Comm com);

/**
 * Parallel training(Slave part) of a given model for the recognition of a set of observation sequences
  * using a modify BaumWelch algorithm based on sequences percentage
 * @param com
 */
void mpiPercentageBaumWelchSlave(MPI_Comm com);

#endif //HMM_MPI_H
