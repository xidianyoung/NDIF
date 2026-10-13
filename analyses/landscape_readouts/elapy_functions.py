# Functions copied from elapy, a Python implementation of the Energy Landscape Analysis Toolbox
# (T. Ezaki et al., Philos. Trans. A Math. Phys. Eng. Sci. 375:20160287, 2017; original MATLAB code
# https://github.com/tkEzaki/energy-landscape-analysis), https://github.com/okumakito/elapy,
# file elapy/core.py.
#
# elapy is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License.  A copy of the License is in LICENSE_elapy.txt in this
# directory and at http://www.apache.org/licenses/LICENSE-2.0.  Unless required by applicable law or
# agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations under the License.
#
# Modification: only the seven functions that the pairwise maximum-entropy readout uses
# (landscape_lib.ising_basins: fit_exact and calc_basin_graph, with the helpers they call) were copied;
# their code is unchanged.
"""Exact pairwise maximum-entropy (Ising) fit and basin graph, from elapy (Apache License 2.0)."""
import numpy as np
import pandas as pd
import networkx as nx

# Notations
# W: weight matrix
# h: external field
# n: number of variables
# k: number of observations
# m: number of all states, 2**n

def calc_state_no(X):
  return X.astype(str).sum().apply(lambda x:int(x,base=2))

def gen_all_state(X_in):
    n = len(X_in)
    X =  np.array([list(bin(i)[2:].rjust(n,'0'))
                for i in range(2**n)]).astype(int).T
    return pd.DataFrame(X, index=X_in.index)

def calc_energy(h, W, X_in):
  X = 2*X_in-1
  return -0.5 * (X * W.dot(X)).sum() - h.dot(X)

def calc_prob(h, W, X):
  energy  = calc_energy(h, W, X)
  energy -= energy.min()  # avoid overflow
  prob    = np.exp(-energy)
  return prob / prob.sum()

# likelihood
def fit_exact(X_in, max_iter=10**4, alpha=0.5):
  X      = 2*X_in-1
  X_all  = gen_all_state(X)
  X2_all = 2*X_all-1
  n, k   = X.shape
  m      = 2**n
  h      = np.zeros(n)
  W      = np.zeros((n,n))
  X_mean = X.mean(axis=1)
  X_corr = X.dot(X.T) / k
  np.fill_diagonal(X_corr.values, 0)
  for i in range(max_iter):
    p      = calc_prob(h, W, X_all)
    Y_mean = X2_all.dot(p)
    Y_corr = X2_all.dot(np.diag(p)).dot(X2_all.T)
    np.fill_diagonal(Y_corr.values, 0)
    h += alpha * (X_mean - Y_mean)
    W += alpha * (X_corr - Y_corr)
    if np.allclose(X_mean, Y_mean) and np.allclose(X_corr, Y_corr):
      break
  return h, W

def calc_adjacent(X):
  X_all = gen_all_state(X)
  out_list = [calc_state_no(X_all)]
  for i in X_all.index:
    Y = X_all.copy()
    Y.loc[i] = 1 - Y.loc[i]
    out_list.append(calc_state_no(Y))
  return pd.concat(out_list, axis=1)

def calc_basin_graph(h, W, X):
  X_all = gen_all_state(X)
  A = calc_adjacent(X)
  energy = calc_energy(h, W, X_all)
  min_idx = energy.values[A].argmin(axis=1)
  graph = pd.DataFrame()
  graph['source'] = A.index.values
  graph['target'] = A.values[A.index, min_idx]
  graph['energy'] = energy
  G = nx.from_pandas_edgelist(graph, create_using=nx.DiGraph)
  graph['state_no'] = None
  conn = sorted(nx.weakly_connected_components(G), key=len)[::-1]
  for i, node_set in enumerate(conn):
    graph.loc[list(node_set),'state_no'] = i + 1
  return graph
