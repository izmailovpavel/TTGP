import numpy as np

def get_x():
  maxlen = 65
  n_features = 6438
  seqs = []
  for i in range(1, 824):
    word_features = np.zeros([maxlen, n_features])
    with open('basenp_novi/'+str(i)+'.x') as file:
      for l in file:
        word, f, _ = l.split('\t')
        word, f = int(word)-1, int(f)
        word_features[word, f] = 1.
      word_features = word_features[:word+1, :]
    seqs.append(word_features)
  return seqs

def get_y():
  maxlen = 65
  n_features = 6438
  labels = []
  for i in range(1, 824):
    seq_labels = []
    with open('basenp_novi/'+str(i)+'.y') as file:
      for l in file:
        label = l[:-1]
        label = int(label)
        seq_labels.append(label)
    labels.append(np.array(seq_labels))
  return labels 

def make_classification_data(x, y):
  x_new, y_new = [], []
  for seq, labels in list(zip(x, y)):
    x_new.append(seq)
    y_new.append(labels)
  x_new = np.concatenate(x_new)
  y_new = np.concatenate(y_new)
  return x_new, y_new

def pad(x, y):	
  maxlen = np.max([seq.shape[0] for seq in x])
  D = x[0].shape[1]
  padded_x = []
  padded_y = []
  seq_lens = []
  for seq, labels in list(zip(x, y)):
    seq_len = seq.shape[0]
    padded_seq = np.concatenate([seq, np.zeros((maxlen - seq_len, D))])[None, :, :]
    padded_labels = np.concatenate([labels, np.zeros((maxlen - seq_len))])[None, :]
    padded_x.append(padded_seq)
    padded_y.append(padded_labels)
    seq_lens.append(seq_len)
  return np.concatenate(padded_x), np.concatenate(padded_y), np.array(seq_lens)



if __name__ =='__main__':
    x = get_x()
    y = get_y()
    x_tr, y_tr = x[:500], y[:500]
    x_te, y_te = x[500:], y[500:]
    x_tr, y_tr = make_classification_data(x_tr, y_tr)
    x_te, y_te = make_classification_data(x_te, y_te)
    
    np.save('data_class/x_tr', x_tr)
    np.save('data_class/x_te', x_te)
    np.save('data_class/y_tr', y_tr)
    np.save('data_class/y_te', y_te)
    
    x = get_x()
    y = get_y()
    x_tr, y_tr = x[:500], y[:500]
    x_te, y_te = x[500:], y[500:]
    
    x_te, y_te, seq_lens_te = pad(x_te, y_te)
    x_tr, y_tr, seq_lens_tr = pad(x_tr, y_tr)
    
    P = np.load('P.npy')
    d, D = P.shape
    x_tr_flat = x_tr.reshape([-1, D])
    x_te_flat = x_te.reshape([-1, D])
    x_tr_flat = x_tr_flat.dot(P.T)
    x_te_flat = x_te_flat.dot(P.T)
    x_tr = x_tr_flat.reshape(list(x_tr.shape[:2])+[d])
    x_te = x_te_flat.reshape(list(x_te.shape[:2])+[d])
    np.save('data_struct/x_tr', x_tr)
    np.save('data_struct/x_te', x_te)
    np.save('data_struct/y_tr', y_tr)
    np.save('data_struct/y_te', y_te)
    np.save('data_struct/seq_lens_tr', seq_lens_tr)
    np.save('data_struct/seq_lens_te', seq_lens_te)
