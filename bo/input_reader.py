import pdb

import os
import re


class ReadInput(object):
    program_path = os.getcwd() + '/'
    ground_path = program_path + '../'

    def __init__(self, input_name):
        com_file = open(input_name + '.com', 'r')
        com_content = com_file.readlines()
        com_file_len = len(com_content)
        com_file.close()

        for line_num in range(com_file_len):
            if com_content[line_num] == 'Model Molecule\n':
                self.model_atom_xyz = []
                self.model_X_atom_xyz = []
                self.model_not_X_atom_xyz = []

                self.total_X_num = 0

                self.atom_mode_X_idx = []
                self.sub_mode_X_idx = []
                self.mol_mode_X_idx = []

                self.part_mode_X_idx_dict = {}
                self.part_root_atom_xyz_dict = {}
                self.root_list_of_each_X = []

                self.X_idx_perm_dict = {}

                self.ori_X_dir_idx_rel = []

                row = 0
                X_idx = 0
                while com_content[line_num + (row + 1)] != 'END\n':
                    com_line = com_content[line_num + (row + 1)].rstrip('\n')
                    com_line = re.split(r'\s+', com_line)

                    atom_xyz = com_line[:4]
                    atom_xyz = [atom_xyz[0]] + [float(i) for i in atom_xyz[1:]]
                    perm = int(com_line[4])
                    part = int(com_line[5])
                    root = int(com_line[6])
                    X_root = int(com_line[7])
                    mode = com_line[8]
                    enum = com_line[9]

                    self.model_atom_xyz.append(atom_xyz)

                    if 'X' in atom_xyz[0]:
                        self.total_X_num += 1
                        self.model_X_atom_xyz.append(atom_xyz)
                        self.root_list_of_each_X.append(X_root)

                        self.X_idx_perm_dict[X_idx] = perm - 1

                        if root != 0:
                            self.part_root_atom_xyz_dict[root] = tuple(atom_xyz)

                        if mode == 'A':
                            self.atom_mode_X_idx.append(X_idx)
                        elif mode == 'S':
                            self.sub_mode_X_idx.append(X_idx)
                        elif mode == 'M':
                            self.mol_mode_X_idx.append(X_idx)

                        if part in self.part_mode_X_idx_dict:
                            self.part_mode_X_idx_dict[part][1].append(X_idx)
                        else:
                            self.part_mode_X_idx_dict[part] = [enum, [X_idx]]

                        X_idx += 1

                    else:
                        self.model_not_X_atom_xyz.append(atom_xyz)
                        if root != 0:
                            self.part_root_atom_xyz_dict[root] = tuple(atom_xyz)

                    row += 1

                for mode, X_idx_list in self.part_mode_X_idx_dict.values():
                    self.ori_X_dir_idx_rel += X_idx_list

        self.locus_cand_name_atom_xyz = [[] for i in range(self.total_X_num)]
        self.locus_cand_num_list = []

        self.cand_name_num_dict = {}
        self.num_cand_name_dict = {}

        self.total_patterns = 1

        sub_idx = -1
        line_num = 0
        cand_num = 0
        while line_num < com_file_len:
            if 'list' in com_content[line_num]:
                sub_idx += 1

                each_locus_cand_num_list = 0
                while com_content[line_num] != 'END\n':
                    each_locus_cand_num_list += 1
                    each_sub_atom_xyz = []
                    each_sub_name = com_content[line_num + 1].rstrip('\n')
                    each_sub_atom_num = com_content[line_num + 2].rstrip('\n')
                    each_sub_atom_num = int(each_sub_atom_num)

                    row = 2
                    for row in range(row, row + each_sub_atom_num):
                        temp_atom_xyz = com_content[line_num + (row + 1)].rstrip('\n')
                        atom_xyz = re.split(r'\s+', temp_atom_xyz)
                        atom_xyz = [atom_xyz[0]] + [float(i) for i in atom_xyz[1:]]
                        atom_xyz = tuple(atom_xyz)

                        each_sub_atom_xyz.append(atom_xyz)
                        row += 1

                    if each_sub_name not in self.cand_name_num_dict:
                        self.cand_name_num_dict[each_sub_name] = cand_num
                        self.num_cand_name_dict[cand_num] = each_sub_name
                        cand_num += 1

                    each_sub_atom_xyz = tuple(each_sub_atom_xyz)
                    self.locus_cand_name_atom_xyz[sub_idx].append((each_sub_name,
                                                                   each_sub_atom_xyz))
                    line_num += row + 1

                self.total_patterns *= each_locus_cand_num_list
                self.locus_cand_num_list.append(each_locus_cand_num_list)

            line_num += 1

        self.locus_cand_name_atom_xyz = \
        tuple([tuple(i) for i in self.locus_cand_name_atom_xyz])

        self.locus_name_num_dict_list = []
        self.locus_gene_list = []

        for X_num in range(self.total_X_num):
            locus_name_num_dict = {}
            locus_gene = []
            for cand_idx in range(self.locus_cand_num_list[X_num]):
                cand_name = self.locus_cand_name_atom_xyz[X_num][cand_idx][0]
                cand_num = self.cand_name_num_dict[cand_name]

                locus_name_num_dict[cand_num] = cand_name
                locus_gene.append(cand_num)

            self.locus_name_num_dict_list.append(locus_name_num_dict)
            self.locus_gene_list.append(locus_gene)
