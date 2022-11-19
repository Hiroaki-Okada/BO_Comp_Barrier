import pdb

import sys
sys.setrecursionlimit(10**9)

from itertools import product
from functools import lru_cache

import numpy as np
import pandas as pd

from bo.input_reader import ReadInput
from bo.model_generator import ModelGenerator
from bo.desc_utils import desc_cleaning, get_desc_info, feature_scaling


class DomainUtils:
    def __init__(self, input_name, desc_data, other_components={}, encoding={}):
        read_input = ReadInput(input_name)
        self.X_num = read_input.total_X_num
        self.locus_gene_list = read_input.locus_gene_list

        self.gen_model = ModelGenerator(input_name)

        self.desc_data = desc_data
        self.other_components = other_components
        self.encoding = encoding

        self.desc_name_dict, self.desc_val_dict = self.read_descriptor()

    def read_descriptor(self):        
        desc_name_dict, desc_val_dict = {}, {}
        for desc_df in self.desc_data:
            desc_df = desc_cleaning(desc_df)
            t_desc_name_dict, t_desc_val_dict = get_desc_info(desc_df)

            desc_name_dict.update(t_desc_name_dict)
            desc_val_dict.update(t_desc_val_dict)

        return desc_name_dict, desc_val_dict

    # @lru_cache(maxsize=None)
    def run(self):
        chem_domain, chem_all_name_combs, chem_columns = self.get_chem_domain()
        other_domain, other_all_name_combs, other_columns = self.get_other_domain()

        if other_domain == []:
            domain = chem_domain
            all_name_combs = chem_all_name_combs
        else:
            domain = list(product(chem_domain, other_domain))
            domain = [sum(i, []) for i in domain]
            all_name_combs = list(product(chem_all_name_combs, other_all_name_combs))
            all_name_combs = [sum(i, []) for i in all_name_combs]

        domain = pd.DataFrame(domain)
        domain.columns = chem_columns + other_columns

        # Feature scaling
        domain = feature_scaling(domain, target=None)

        all_name_combs = pd.DataFrame(all_name_combs)
        col = ['X' + str(i + 1) for i in range(self.X_num)]
        col += [i for i in self.other_components.keys()]
        all_name_combs.columns = col

        return domain, all_name_combs

    # Domain construction of chemical species
    def get_chem_domain(self):
        self.seen = set()
        self.chem_domain = []
        self.chem_all_name_combs = []

        self.dfs([], -1)

        chem_columns = []
        name_comb_example = self.chem_all_name_combs[0]
        for X_num in range(1, self.X_num + 1):
            name = name_comb_example[X_num - 1]
            chem_columns += ['X' + str(X_num) + '_' + i for i in self.desc_name_dict[name]]

        self.chem_all_name_combs = [list(i) for i in self.chem_all_name_combs]

        return self.chem_domain, self.chem_all_name_combs, chem_columns

    def dfs(self, cand_num_comb_l, sub_idx):
        if len(cand_num_comb_l) == self.X_num:
            cand_name_comb = self.gen_model.get_ini_cand_name_comb(cand_num_comb_l)

            _, c_cand_name_comb, isDuplicate = \
            self.gen_model.duplicate_check(cand_name_comb, self.seen)

            if not isDuplicate:
                self.seen.add(c_cand_name_comb)
                self.expand_domain(c_cand_name_comb)

            return

        sub_idx += 1
        for cand_num in self.locus_gene_list[sub_idx]:
            self.dfs(cand_num_comb_l + [cand_num], sub_idx)

    def expand_domain(self, c_cand_name_comb):
        self.chem_all_name_combs.append(c_cand_name_comb)

        desc_comb = []
        for cand_name in c_cand_name_comb:
            cand_desc = self.desc_val_dict[cand_name]
            desc_comb += cand_desc

        self.chem_domain.append(desc_comb)

    # Domain construction of numeric & ohe inputs
    def get_other_domain(self):
        other_columns = []
        other_domain = []
        other_all_name_combs = []

        for key in self.other_components:
            possible = self.other_components[key]

            if key in self.encoding:
                encoding_type = self.encoding[key]
            else:
                encoding_type = 'ohe'

            if encoding_type.lower() == 'numeric':
                 tmp_columns = [key]
                 tmp_domain = [[i] for i in possible]
                 tmp_name_comb = [[i] for i in possible]
            if encoding_type.lower() == 'ohe':
                 tmp_columns = [key + '=' + str(i) for i in possible]
                 tmp_domain, tmp_name_comb = self.one_hot_encode(key, possible)

            other_columns += tmp_columns

            if other_domain == []:
                other_domain = tmp_domain
                other_all_name_combs = tmp_name_comb
            else:
                domain_product = product(other_domain, tmp_domain)
                name_comb_product = product(other_all_name_combs, tmp_name_comb)
                other_domain = [sum(i, []) for i in domain_product] 
                other_all_name_combs = [sum(i, []) for i in name_comb_product]

        return other_domain, other_all_name_combs, other_columns

    def one_hot_encode(self, key, possible):
        ohe = []
        name_comb = []
        for val in possible:
            row = self.one_hot_row(val, possible)
            ohe.append(row)
            name_comb.append([val])

        return ohe, name_comb

    def one_hot_row(self, val, possible):
        row = []
        for entry in possible:
            if entry == val:
                row.append(1)
            else:
                row.append(0)

        return row
