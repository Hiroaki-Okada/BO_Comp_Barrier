from bo.input_reader import ReadInput
from bo.duplicate_checker import DuplicateChecker


class ModelGenerator(ReadInput):
    def __init__(self, input_name):
        super().__init__(input_name)
        self.check_duplicate = DuplicateChecker(input_name)

    def get_ini_cand_name_comb(self, individual):
        cand_name_comb = []
        for gene in individual:
            cand_name = self.num_cand_name_dict[gene]
            cand_name_comb.append(cand_name)

        return cand_name_comb

    def duplicate_check(self, cand_name_comb, seen=set()):
        c_dir_name, c_cand_name_comb, isDuplicate = self.check_duplicate.run(cand_name_comb, seen)
        return c_dir_name, c_cand_name_comb, isDuplicate
