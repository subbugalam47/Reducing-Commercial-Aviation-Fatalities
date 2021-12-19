# Reducing-Commercial-Aviation-Fatalities

## Introduction:

My problem statement was a kaggle competition which is organized by Booz Allen Hamilton company on Kaggle. Booz Allen Hamilton has been solving for business, government, and military leaders for over 100 years. Most flite related fatalities originate from lack of airplane state awareness.That implies ineffective attention management on part of pilots who may be distracted , sleepy or in other dangerous cognitive states.Our challenge is to build a model to detect troubling events from aircrew’s physiological data

## Dataset features Analysis:

I want separate features as into general and sensor based features for convenience. Given dataset consist 6 general features like id,experiment,crew,time, seat,event and 23 sensor based recordings among them 20 brain activities signal recordings ,1 Electrocardiogram signal recordings,1 Respiration sensor recordings ,1 Galvanic Skin Response sensor

### General features

id - A unique identifier for a crew + time combination. You must predict probabilities for each id.

crew - a unique id for a pair of pilots. There are 9 crews in the data. experiment - One of CA, DA, SS or LOFT. The first 3 comprise the training set. The latter is the test set.

time - seconds into the experiment

event - The state of the pilot at the given time: one of A = baseline, B = SS, C = CA, D = DA

seat - is the pilot in the left (0) or right (1) seat

### Sensors based features

eeg prefix are electroencephalogram recordings(brain activity recordings)

eeg_fp1,eeg_f7,eeg_f8,eeg_t4,eeg_t6,eeg_t5,eeg_t3,eeg_fp2,eeg_o1,eeg_p3,eeg_pz,eeg_f3,eeg_fz,eeg_f4,eeg_c4,eeg_p4,eeg_poz,eeg_c3,Eeg_cz,eeg_o2

ecg - Electrocardiogram signal. The sensor had a resolution/bit of .012215 µV and a range of -100mV to +100mV. The data are provided in microvolts.

r - Respiration, a measure of the rise and fall of the chest. The sensor had a resolution/bit of .2384186 µV and a range of -2.0V to +2.0V. The data are provided in microvolts.

gsr - Galvanic Skin Response, a measure of electrodermal activity. The sensor had a resolution/bit of .2384186 µV and a range of -2.0V to +2.0V. The data are provided in microvolts.

## Business-Problem:

We are provided with real physiological data from eighteen pilots who were subjected to various distracting events. The benchmark training set is a set of controlled experiments collected in a non-flight environment, outside of a flight simulator. The test set (abbreviated LOFT = Line Oriented Flight Training) consists of a full flight (take off, flight, and landing) in a flight simulator

Pilots experienced distractions intended to introduce one of following 3 cognitive states

**Channelized Attention (CA)** is, roughly speaking, the state of being focused on one task to the exclusion of all others. This is induced in benchmarking by having the subjects play an engaging puzzle-based video game.

**Diverted Attention (DA)** is the state of having one’s attention diverted by actions or thought processes associated with a decision. This is induced by having the subjects perform a display monitoring task. Periodically, a math problem showed up which had to be solved before returning to the monitoring task.

**Startle/Surprise (SS)** is induced by having the subjects watch movie clips with jump scares. Our aim is to build a model that can predict the state of the mind of the pilot in real time using the given psychological data by running calculations in real time to monitor the cognitive states of pilots which could help pilots to be alert when they enter a trouble state so that we can prevent accidents

## ML Formulation of Business Problem

This is a Multiclass classification(A,B,C,D) problem. For each id , we need to predict the probability of each state of the pilot at the given time. one of A = baseline, B = SS, C = CA, D = DA
