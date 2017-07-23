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
