import numpy as np

class TransformerBlock:
  def __init__(self, id, dim, num_heads):
    self.cached = {}
    self.id = id
    self.dim = dim
    self.num_heads = num_heads
  
  def forward(self, X, analyzer):
    cached = self.cached

    cached['x'] = np.array(X)

    attn, A_list, Q_list, K_list, V_list = analyzer.multi_head_attention(X, W_q_heads=cached['W_q_heads'], 
                                          W_k_heads=cached['W_k_heads'], W_v_heads=cached['W_v_heads'],
                                          W_o=cached['W_o'])
    
    cached['attn'] = attn
    cached['A_list'] = A_list
    cached['Q_list'] = Q_list
    cached['K_list'] = K_list
    cached['V_list'] = V_list

    attn_norm = analyzer.LayerNorm(attn, gamma=np.ones_like(attn[0]), beta=np.zeros_like(attn[0]))
    cached['attn_norm'] = attn_norm
    res1 = analyzer.add_substep(X, attn_norm)
    cached['res1'] = res1

    ln1 = analyzer.LayerNorm(res1, gamma=cached['gamma1'], beta=cached['beta1'])
    cached['ln1'] = ln1

    ffn_out, a1, h1 = analyzer.ffn(ln1, W_1=cached['W_1'], W_2=cached['W_2'], b_1=cached['b_1'], b_2=cached['b_2'])
    cached['ffn_out'] = ffn_out
    cached['a1'] = a1
    cached['h1'] = h1

    res2 = analyzer.add_substep(ln1, ffn_out)
    cached['res2'] = res2

    ln2 = analyzer.LayerNorm(res2, gamma=cached['gamma2'], beta=cached['beta2'])
    cached['ln2'] = ln2

    return ln2