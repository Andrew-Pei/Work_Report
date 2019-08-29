Weekly work report
==================

After the meeting last week, I have been engaged in experiments. I have tried
the following:

1.  I have rewritten many official examples, but many of the examples (agnn,
    gat, gcn, cora, etc.) have not matched the topk data format, and there is no
    parameter ”batch“. So this method did not succeed in most cases. Only the same
    parameters are found in example triangles_sag_pool.py.

2.  I found that the topk pooling method can be applied to
    triangles_sag_pool.py. So, I rewrote this code. However, after I rewrote,
    during the running of the code, I found that the code was running very
    slowly, and it took nearly 30 seconds to complete an epoch. And the final
    accuracy is very poor, whether using sagpooling or topkpooling, the accuracy
    rate is only 55%.

Future plan:
------------

1.  Try to fix the problem of triangles_sag_pool.py

2.  Try the other datasets used in the following papers and the code in the
    paper:

[Graph U-Nets](https://arxiv.org/pdf/1905.05178)

[Towards Sparse Hierarchical Graph Classifiers](https://arxiv.org/abs/1811.01287)

[Understanding Attention and Generalization in Graph Neural Networks](https://arxiv.org/abs/1905.02850)