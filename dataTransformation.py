import os, random, math, copy
import numpy as np
import dill as pk
from sklearn.model_selection import train_test_split


def convert_to_one_hot_vector(samples, num_labels=None):
    #convert labels into one hot vector, assumes no gap in the labels
    tmp = np.array(samples)
    if (num_labels is None) or (num_labels < tmp.max()+1):
        max_lbls = tmp.max()+1
    else:
        max_lbls = num_labels
    one_hot_labels = np.zeros((tmp.size, max_lbls))
    one_hot_labels[np.arange(tmp.size),tmp] = 1
    return one_hot_labels


def load_data_label(pk_pth):
    data = pk.load(open(pk_pth,'rb'))
    X = data['data']
    y = data['label']
    return X, y


def load_split_data(pk_pth, random_seed=None):
    X, y = load_data_label(pk_pth)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=random_seed) #split evenly
    return X_train, y_train, X_test, y_test


def distribute_classification_data(X_data, y_label, num_clients, uneven_ratio = None, random_seed = True):
    #distribute data to each client
    def _reset_seeds(seed=42):
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _shuffle_data(data_list, label_list):
        for cli in range(len(data_list)):
            p = np.random.permutation(data_list[cli].shape[0])
            data_list[cli] = data_list[cli][p]
            label_list[cli] = label_list[cli][p]
        return data_list, label_list   
        
    if random_seed is None:
        pass
    elif random_seed == True:
        _reset_seeds()
    elif isinstance(random_seed,int):
        _reset_seeds(random_seed)
        
    data_lst = []
    label_lst = []
    
    labels = np.unique(y_label)
    num_labels = len(labels)
    n_samples_cli = int(len(y_label)//num_clients)
    n_samp_cli_class  = int(n_samples_cli//num_labels)
    cls_ind_lst = []
    for i in labels:
        ind = np.where(y_label==i)
        cls_ind_lst.append(np.random.permutation(ind[0]))
    data_ind = [0]*num_labels
    for i in range(num_clients):
        cli_data = []
        cli_label = []
        for ind, cls_ind in enumerate(cls_ind_lst):
            if uneven_ratio is None:
                cli_data.append(X_data[cls_ind[(i*n_samp_cli_class): (i*n_samp_cli_class)+ n_samp_cli_class]])
                cli_label.append(y_label[cls_ind[(i*n_samp_cli_class): (i*n_samp_cli_class)+ n_samp_cli_class]])
            else:
                if ind==(i%num_labels):
                    n = int(n_samples_cli*uneven_ratio)
                else:
                    n = int((n_samples_cli*(1-uneven_ratio))//(num_labels-1))
                cli_data.append(X_data[cls_ind[data_ind[ind]: data_ind[ind]+ n]])
                cli_label.append(y_label[cls_ind[data_ind[ind]: data_ind[ind]+ n]])
                data_ind[ind] += n
        data_lst.append(np.concatenate(cli_data,axis=0))
        label_lst.append(np.concatenate(cli_label,axis=0))
    data_lst, label_lst = _shuffle_data(data_lst, label_lst)
    return data_lst, label_lst


def distribute_classification_data_wrapper(data, one_hot_lab, cli_per_data, uneven_ratio = None, random_seed = True):
    #takes in one-hot labels instead of int labels
    data_lst, label_lst = distribute_classification_data(data,np.argmax(one_hot_lab,axis=1), cli_per_data, uneven_ratio=uneven_ratio, random_seed=random_seed)
    num_lbl = one_hot_lab.shape[-1]
    label_lst = [convert_to_one_hot_vector(i, num_lbl) for i in label_lst]
    return data_lst, label_lst


def labels4clients(tot_labels, num_labels, tot_clients, unique_clients, random_seed = True):
    # This will assign each client with num_labels
    # - tot_labels: the number of total labels you want to consider
    # - num_labels: the number of labels each client should have, required to be less than or equal to tot_labels
    # - tot_clients: how many total clients
    # - unique_clients: the number of unique combinations should be generated. Requirement: N*unique_clients = tot_clients (where N is a positive integer)
    # Output will be a dict of label-client list pairs
    assert num_labels <= tot_labels, 'Requirement: num_labels <= tot_labels'
    assert isinstance(tot_labels, int) and isinstance(num_labels, int) and isinstance(tot_clients, int) and isinstance(unique_clients, int), 'Inputs should be positive integers'
    assert (tot_labels > 0) and (num_labels > 0) and (tot_clients > 0) and (unique_clients > 0), 'Inputs should be positive integers'
    assert tot_clients%unique_clients==0, 'tot_clients needs to be a factor of unique_clients'
    
    def _reset_seeds(seed=42):
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _check_duplicate(ddict, sset): #check if set already exists in dict in some (key-)value (pair)
        for key in ddict:
            if sset == ddict[key]:
                return True
        return False

    factor = tot_clients//unique_clients
    veh_lbl_dict = dict()
    
    if tot_labels == num_labels:
        for k in range(unique_clients):
            veh_lbl_dict[k] = [n for n in range(num_labels)]
    else:
        #assert math.comb(tot_labels,num_labels) >= unique_clients, 'unique_clients cannot exceed the number of possible combinations of "tot_labels choose num_labels" (n choose k)'

        if random_seed==True:
            pass
        elif isinstance(random_seed,int):
            _reset_seeds(random_seed)
        else:
            _reset_seeds()

        all_lbls = np.arange(tot_labels).tolist()
        tmp_lbls = np.arange(tot_labels).tolist()

        for n in range(unique_clients):
            vehlbl = set()
            if not(tmp_lbls): #if not empty
                tmp_lbls = np.arange(tot_labels).tolist()

            check = True
            while (len(vehlbl) < num_labels) or (_check_duplicate(veh_lbl_dict, vehlbl)):# or not(veh_lbl_dict)):
                np.random.shuffle(tmp_lbls) #shuffle list
                if (num_labels <= len(tmp_lbls)) and check: #add all the labels
                    vehlbl = set(tmp_lbls[:num_labels])
                elif (num_labels > len(tmp_lbls)) and check: #add remaining labels then let it run until a unique permutation is found (going into else statement)
                    vehlbl = set(tmp_lbls)
                    keep_vehlbl = copy.deepcopy(vehlbl)
                    tmp_lbls = list(set(all_lbls) - vehlbl) #only consider the missing labels
                    check = False
                else:
                    if len(vehlbl) == num_labels: #reset if no unique permutation found
                        vehlbl = copy.deepcopy(keep_vehlbl)
                    else:
                        vehlbl.add(tmp_lbls[0])
            if check:
                tmp_lbls = list(set(tmp_lbls) - vehlbl) #remove the labels that got added into vehlbl
            else:
                tmp_lbls = list(set(all_lbls) - (vehlbl - keep_vehlbl)) #remove only new labels added

            veh_lbl_dict[n] = vehlbl
    
        for k in veh_lbl_dict: #make value from set to list and sort
            veh_lbl_dict[k] = list(veh_lbl_dict[k])
            veh_lbl_dict[k].sort()
    
    #duplicate to other clients if tot_clients > unique_clients
    for m in range(1,factor): 
        for n in range(unique_clients):
            veh_lbl_dict[n+m*unique_clients] = copy.deepcopy(veh_lbl_dict[n])
    
    #inverse the veh_lbl_dict (so it's label-vehicle key-value pairs)
    lbl_veh_dict = dict()
    for k in range(tot_labels):
        lbl_veh_dict[k] = []
    for key in veh_lbl_dict:
        for el in veh_lbl_dict[key]:
            lbl_veh_dict[el].append(key)
            
    return lbl_veh_dict


def distribute_data_labels4clients(X_data, y_label, labels4clients_dict, random_seed = True, max_samples_per_client = None):
    #distribute data to each client according to the dict obtain from labels4clients function
    #max_samples_per_client is an integer for the number of samples each client should have
    
    def _reset_seeds(seed=42):
        os.environ['PYTHONHASHSEED']=str(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    def _cli_in_dict(labels4clients_dict): #get the client numbers
        lst = []
        for el in labels4clients_dict:
            lst.extend(labels4clients_dict[el])
        return np.unique(np.array(lst))
    
    def _shuffle_data(data_list, label_list):
        for cli in range(len(data_list)):
            p = np.random.permutation(data_list[cli].shape[0])
            data_list[cli] = data_list[cli][p]
            label_list[cli] = label_list[cli][p]
        return data_list, label_list    
    
    labels = np.unique(y_label)
    num_labels = len(labels)
    dict_keys = list(labels4clients_dict.keys())
    
    assert set(labels).issuperset(set(dict_keys)), 'The label keys in labels4clients_dict are not a subset of y_label'
    
    if random_seed==True:
        pass
    elif isinstance(random_seed,int):
        _reset_seeds(random_seed)
    else:
        _reset_seeds()
    
    n_samp_per_lbl = int(len(y_label)//num_labels) #assumes same amount of samples per label
    n_cli_per_lbl = len(labels4clients_dict[dict_keys[0]]) #assumes same amount of clients per label
    n_samp_per_lbl_cli = int(n_samp_per_lbl//n_cli_per_lbl)
    if isinstance(max_samples_per_client,int):
        n_samp_per_lbl_cli = min(n_samp_per_lbl_cli, max_samples_per_client)
    avail_clients = _cli_in_dict(labels4clients_dict)
    
    cli_data_dict = dict()
    cli_lbl_dict = dict()
    for el in avail_clients:
        cli_data_dict[el] = []
        cli_lbl_dict[el] = []
    
    for lbl in labels4clients_dict:
        ind = np.random.permutation(np.where(y_label==lbl)[0])
        for k, cli in enumerate(labels4clients_dict[lbl]):
            cli_data_dict[cli].extend(X_data[ind[k*n_samp_per_lbl_cli:(k+1)*n_samp_per_lbl_cli]])
            cli_lbl_dict[cli].extend(y_label[ind[k*n_samp_per_lbl_cli:(k+1)*n_samp_per_lbl_cli]])
    
    data_lst = [np.array(cli_data_dict[el]) for el in cli_data_dict]
    label_lst = [np.array(cli_lbl_dict[el]) for el in cli_lbl_dict]
    
    data_lst, label_lst = _shuffle_data(data_lst, label_lst)
    
    return data_lst, label_lst
    

def distribute_data_labels4clients_wrapper(data, one_hot_lab, labels4clients_dict, random_seed = True, max_samples_per_client = None):
    #takes in one-hot labels instead of int labels
    data_lst, label_lst = distribute_data_labels4clients(data,np.argmax(one_hot_lab,axis=1), labels4clients_dict, random_seed=random_seed, max_samples_per_client=max_samples_per_client)
    num_lbl = one_hot_lab.shape[-1]
    label_lst = [convert_to_one_hot_vector(i, num_lbl) for i in label_lst]
    return data_lst, label_lst
    
    
def make_uni_dis(X, y, n=None):
    new_X = None
    new_y = None
    if n is None:
        n = np.min([len(np.where(y==i)[0]) for i in np.unique(y)])
    for i in np.unique(y):
        #ind = np.where(y==i)[0]
        #ind = np.random.permutation(ind)[:n]
        ind = np.where(y==i)[0][:n]
        if new_X is None and new_y is None:
            new_X = X[ind]
            new_y = y[ind]
        else:
            new_X  = np.concatenate((new_X,X[ind]),axis=0) 
            new_y  = np.concatenate((new_y,y[ind]),axis=0) 
    return new_X, new_y