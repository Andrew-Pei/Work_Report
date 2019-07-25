# Work report:

This week, I mainly worked on clarifying concepts and reading literature. What I did could be divided into 2 parts: make a foundation for studying graph pooling and investigate graph pooling.

The notes for this part of my study are as follows, a more readable version of PDF is in the attachment.

## GNN foundation:

- papers

  - website

  - review
    - strongly recommended
      - review paper
       	    <details>
       	            <summary>problem in reading</summary>
       	                  	diffusion process<br>
       	                  	what is a parameterization with smooth coefficients<br>
       	                  	K-localized
       	                </details>
      
      - notes
      
        - key distinction method: recognizing corresponding aggregators and updaters
      - RNN kernel: GRU, LSTM
        - aggregator and updater: The variants utilize different **aggregators** to gather information from each node's neighbors and specific **updaters** to update nodes' hidden states.
    
  - models
  
    - training methods
      - the first paper
        - topic: flat—\>hierarchy
        - problems in reading
        - differentiable: 可区分？可微分？√
        - pros: small time complexity
        - cons: non-convex, unstable
    
  - applications
  
    - link prediction: based on time-stamped data
    - node classification: just assign a number(or multiple number) to each node
    - graph classification
  
- New knowledge
  - ？Gate mechanism
  - ？Skip connection mechanism
  - ？Attention mechanism
  - ？Dropout mechanism
  - graph embedding
    - vertex embedding
      - DeepWalk
        - word2vec
        - skip-gram
        - pros
        - cons: It lacks the ability of generalization. Whenever a new node comes in, it has to re-train the model in order to represent this node (transductive). Thus, such GNN is not suitable for dynamic graphs where the nodes in the graphs are ever-changing
      - node2vec
      - SDNE
- graph embedding
      - graph2vec

- Reviewed knowledge

  - CNN

    - convolution

    - pooling

      - Pooling classification_1

        - general pooling
          - max pooling
          - average pooling

        - overlap pooling

        - spatial pyramid pooling

      - Pooling classification_2
        - maxPooling over time
        - k-max pooling
        - chunk-max pooling

    - feature
      - Shared weight
      - Local connectivity
      - 3D volumes of neurons

- problem
  - message
  - basic pooling process
  - fixed point theorem
  - what is compositionality in CNN
    - Answer：Simoncelli & Olshausen, 2001

- Application
  - Modeling physical systems
  - Learn molecular footprint
  - Predict protein interface
  - Classify disease

## Graph pooling:

<details>
    <summary>history</summary>
    Previous researches have adopted the pooling method that considers only graph topology (Defferrard et al., 2016; Rhee et al., 2018). With growing interest in graph pooling, several improved methods have been proposed (Dai et al., 2016; Duvenaud et al., 2015; Gilmer et al., 2017b; Zhang et al., 2018b). They utilize node features to obtain a smaller graph representation. Recently, Ying et al.; Gao & Ji; Cangea et al. have proposed innovative pooling methods that can learn hierarchical representations of graphs. These methods allow Graph Neural Networks (GNNs) to attain scaled-down graphs after pooling in an end-to-end fashion.
</details>

- Classification
  - topology based pooling
    - pros
    - cons
      - time complexity
    
  - global pooling
  
  - hierarchical pooling
  
    - DiffPool
      - code available
      - storage complexity
  
    - gPool
      - storage complexity

- temp

  - MixHop
    - allows full linear mixing of neighborhood information
    - objective
    - problem in reading
      - ？Delta Operators
    
    - Unlearned prior knowledge
      - GCN
  
  - SAGPooling
    - Unlearned prior knowledge
      - Self-attention mask Attention mechanisms
      - graph convolution
    - key improvement compared with gPool
      - SAGPooling uses  the ﬁrst order approximation of the graph Laplacian to calculate the attention scores of nodes. Whose performance is improved.
    - pros
      - It can be both H architecture and G architecture. Therefore, it can be used to learn both many vertexes and few vertexes graph. 
      - (Don't quite understand): 5.4. Relation with the Number of Nodes
      - reasonable complexity
      - end-to-end representation learning
    - cons
      - 5.6. Limitations
# Plans of next week:

Next week, I hope to get started with mathematics derivation and code reading as soon as possible.

1.  Analyze algorithm complexity and implemention overhead

2.  Understand the code

3.  Analyze the pros and cons of each method
