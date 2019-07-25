### Work report:

This week, I mainly worked on clarifying concepts and reading literature. What I
did could be divided into 2 parts: make a foundation for studying graph pooling
and investigate graph pooling.

The notes for this part of my study are as follows, a more readable version of
pdf is in the attachment.

1.  GNN foundation:

papers

website

review

strongly recommended

revier paper

problem in reading

diffusion process

what is a parameterization with smooth coefficients

K-localized

a problem

notes

key distinction method

recognizing corresponding aggregators and updaters

RNN kernel

GRU

LSTM

aggregator and updater

propagation types

Convolution

knowledge point

Symmetric normalized Laplacian

models

training methods

the first paper

topic：flat—\>hierarchy

problems in reading

differentiable

可区分？

可微分？√

pros

small time complexity

cons

non-convex

unstable

applications

New knowledge

？Gate mechanism

？Skip connection mechanism

？Attention mechanism

？Dropout mechanism

graph embedding

vertex embedding

DeepWalk

word2vec

skip-gram

pros

cons

node2vec

SDNE

graph embedding

graph2vec

Reviewed knowledge

CNN

conv

pooling

Pooling classification1

general pooling

max pooling

average pooling

overlap pooling

spatial pyramid pooling

Pooling classification2

maxPooling over time

k-max pooling

chunk-max pooling

feature

Shared weights

Local connectivity

3D volumes of neurons

SVM

Data structure

graph

problem

message

basic pooling process

fixed point theorem

what is compositionality in CNN

answer：Simoncelli & Olshausen, 2001

Application

Modeling physical systems

Learn molecular footprint

Predict protein interface

Classify disease

1.  Graph pooling:

history

Classification

topology based pooling

pros

cons

time complexity

global pooling

hierarchical pooling

DiffPool

code available

storage complexity

gPool

storage complexity

temp

MixHop

allows full linear mixing of neighborhood information

objective

problem in reading

？Delta Operators

Unlearned prior knowledge

GCN

SAGPooling

Unlearned prior knowledge

Self-attention mask Attention mechanisms

graph convolution

### Plans of next week:

Next week, I hope to get started with mathematics derivation and code reading as
soon as possible.

1.  Analyze algorithm complexity and implementaion overhead

2.  Understand the code

3.  Analyze the pros and cons of each method
