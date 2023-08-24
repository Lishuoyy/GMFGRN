import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description="")

parser.add_argument('-save_dir', default=None, required=False, help='output file path; use current path if None')
parser.add_argument('-out_filename', required=True, help='output file name')
parser.add_argument('-pos_filepath', required=True, help='TF pos file path')
parser.add_argument('-n_folds', default=3, type=int,help='number of folds to be divided.')

args = parser.parse_args()


n_fold = args.n_folds
save_dir = args.save_dir
out_filename = args.out_filename
pos_filepath = args.pos_filepath

if save_dir == None:
    save_dir = os.getcwd()


## read pos file
s = open(pos_filepath,'r')
pos_list = []
for line in s:
    read = line.split()[0]
    pos_list.append(read)
s.close()
    
for i in range(len(pos_list)):
    pos_list[i] = int(pos_list[i])


## trans pos information to n_sample
num_list = []
for i in range(len(pos_list)-1):
    num_list.append(pos_list[i+1]-pos_list[i])

num_list_sort = num_list
num_list_sort.sort(reverse=True)


## average group(fold)
balance_groups = [[] for i in range(n_fold)]
for v in num_list_sort:
    balance_groups.sort(key=lambda x: sum(x))
    balance_groups[0].append(v)
print(balance_groups)

balance_groups_index = [[] for i in range(n_fold)] 
for i in range(n_fold):
    balance_groups_index[i] = np.where(np.in1d(num_list,balance_groups[i]))[0].tolist()

print(balance_groups_index)
exit()
## save TF index    
save_file = open(os.path.join(save_dir, out_filename + '.txt'),'w')

for i in range(n_fold):
    for j in range(len(balance_groups_index[i])):
        save_file.write(str(balance_groups_index[i][j]))
        if j < len(balance_groups_index[i])-1:
            save_file.write(',')
    save_file.write('\n')
save_file.close()


# python fold_divide_ByTF.py -pos_filepath ./DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt -save_dir /home/comp/csyuxu/dynDeepDRIM -out_filename hesc1_cross_validation_fold_divide -n_folds 3
