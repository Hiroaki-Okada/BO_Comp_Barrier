import os
import sys
sys.setrecursionlimit(10**9)

from bo.input_reader import ReadInput


class DuplicateChecker(ReadInput):
    def __init__(self, input_name):
        super().__init__(input_name)

    def run(self, cand_name_comb, seen=set()):
        self.seen = seen
        self.part_X_name_enum_l = self.get_name_enum(cand_name_comb)

        self.dir_name_l = []
        self.cand_name_comb_l = []
        self.isDuplicate_l = []

        self.dfs_enumeration([])

        if any(self.isDuplicate_l):
            idx = self.isDuplicate_l.index(True)
            c_dir_name = self.dir_name_l[idx]
            c_cand_name_comb = self.cand_name_comb_l[idx]
            isDuplicate = True
        else:
            c_dir_name = self.dir_name_l[0]
            c_cand_name_comb = self.cand_name_comb_l[0]
            isDuplicate = False

        c_cand_name_comb = tuple(c_cand_name_comb)

        return c_dir_name, c_cand_name_comb, isDuplicate

    def get_name_enum(self, cand_name_comb):
        part_X_name_enum_l = []
        for mode, X_idx in self.part_mode_X_idx_dict.values():
            name_l = []
            for each_X_idx in X_idx:
                name_l.append(cand_name_comb[each_X_idx])

            each_part_X_name_enum_l = []
            if mode == 'P':
                each_part_X_name_enum_l.append(name_l)
            elif mode == 'B':
                each_part_X_name_enum_l.append(sorted(name_l))
            elif mode == 'C':
                for i in range(len(name_l)):
                    temp_name_l = name_l[i:] + name_l[:i]
                    if temp_name_l not in each_part_X_name_enum_l:
                        each_part_X_name_enum_l.append(temp_name_l)

            part_X_name_enum_l.append(each_part_X_name_enum_l)

        return part_X_name_enum_l

    def dfs_enumeration(self, temp_name_enum):
        if len(temp_name_enum) == len(self.part_X_name_enum_l):
            name_enum_sum = sum(temp_name_enum, [])

            new_cand_name_comb = [''] * self.total_X_num
            for cand_name, dir_idx in zip(name_enum_sum, self.ori_X_dir_idx_rel):
                new_cand_name_comb[dir_idx] = cand_name

            X_num = 1
            dir_name = ''
            while X_num <= self.total_X_num:
                dir_name += 'X' + str(X_num) + '-' + new_cand_name_comb[X_num - 1]
                if X_num < self.total_X_num:
                    dir_name += '_'

                X_num += 1

            self.dir_name_l.append(dir_name)
            self.cand_name_comb_l.append(new_cand_name_comb)

            if dir_name in self.seen:
                self.isDuplicate_l.append(True)
            else:
                self.isDuplicate_l.append(False)

            return

        next_idx = len(temp_name_enum)
        for cand_name in self.part_X_name_enum_l[next_idx]:
            self.dfs_enumeration(temp_name_enum + [cand_name])
