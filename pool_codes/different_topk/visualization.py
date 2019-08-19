G = nx.Graph()
num_nodes = scatter_add(batch.new_ones(score.size(0)), batch, dim=0)
G.add_nodes_from(np.arange(num_nodes))
loop_length = edge_index.size(1)
edge_index_np = edge_index.numpy()
edge_index_np = np.transpose(edge_index_np)
G.add_edges_from(edge_index_np)
nx.draw(G, with_labels=True, font_weight='bold')

new_nodes = torch.LongTensor(range(num_nodes))[perm]
place = new_nodes[0]

new_edge_index = np.arange(loop_length*2).reshape(edge_index.size(0), edge_index.size(1))
for i in range(loop_length):
	if edge_index[0][i] not in new_nodes or edge_index[1][i] not in new_nodes:
		new_edge_index[0][i] = place
		new_edge_index[1][i] = place
	else:
		new_edge_index[0][i] = edge_index[0][i]
		new_edge_index[1][i] = edge_index[1][i]

G1 = nx.Graph()
G1.add_nodes_from(new_nodes)
new_edge_index = np.transpose(new_edge_index)
G1.add_edges_from(new_edge_index)
nx.draw(G1, with_labels=True, font_weight='bold')
plt.show()
plt.savefig("/home/bowen/zheshiwode.jpg")