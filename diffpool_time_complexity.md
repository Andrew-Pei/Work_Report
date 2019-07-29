# Time complexity:

According to the code and paper, I calculated the time complexity of DiffPooling: 

if we consider the l-th layer in the GNN, then the operation includes:


| left | right |
| :--: | :---: |
|k | pooling ratio|
|L|total number of layers|
|l|l-th layer|
<center>Table 1. Notations</center>
1. BatchedGraphSAGE model: 

   code:

   ```python
   self.embed = BatchedGraphSAGE(nfeat, nhid, device=self.device, use_bn=True)
   self.assign_mat = BatchedGraphSAGE(nfeat, nnext, device=self.device, use_bn=True)
   ...
   z_l = self.embed(x, adj)
   s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
   ```



Although the BG(BatchedGraphSAGE) model is part of the pooling, there are other options besides the BG model, and the BG model is not the optimization object of this paper, so the time complexity of the BG model is not calculated and analyzed.

2. Softmax operation:

   code:
   
```python
s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
```

The softmax operation includes exponentiation, addition, and division. Time complexity is

<img src="http://latex.codecogs.com/gif.latex?O(n_{l}\times n_{l+1}+n_{l}\times n_{l+1}+n_{l})=O(k\times n_{l}^{2})" />

3. Matrix multiplication

   code:

```python
xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
...
self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
```
<img src="http://latex.codecogs.com/gif.latex?O(n_{l+1}\times n_{l}\times d+3\times n_{l}^{2}\times n_{l+1})=O(k\times n_{l}^{2}\times d+k\times n_{l}^{3})" />

4. Matrix subtraction

   code:
   
```python
self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
```

<img src="http://latex.codecogs.com/gif.latex?O(n_{l}\times n_{l})=O(n_{l}^{2})" />

5. Frobenius norm

   code:
```python
self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
```

<img src="http://latex.codecogs.com/gif.latex?O(n_{l}\times n_{l})=O(n_{l}^{2})" />

6. entropy function
   
   code:
   
```python
self.entropy_loss = torch.distributions.Categorical(probs=s_l).entropy()
if mask is not None:
	self.entropy_loss = self.entropy_loss * mask.expand_as(self.entropy_loss)
self.entropy_loss = self.entropy_loss.sum(-1)
```

<img src="http://latex.codecogs.com/gif.latex?O(n_{l+1}\times n_{l+1})=O(n_{l+1}^{2})" />

### Conclusion

   The time complexity of DiffPool is <img src="http://latex.codecogs.com/gif.latex?O(n_{l+1}\times n_{l}\times d+3\times n_{l}^{2}\times n_{l+1})=O(k\times n_{l}^{2}\times d+k\times n_{l}^{3})" />
