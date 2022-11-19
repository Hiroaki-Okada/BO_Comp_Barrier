import pandas as pd

from bo.objective import Objective


class BOUtils:
    def get_kernel(self):
        num_dims = 0 if len(self.obj.X) == 0 else len(self.obj.X[0])
        self.kernel_utils.num_dims = num_dims

        self.kernel_utils.opt_kernel(self.obj.X, self.obj.y)

        return self.kernel_utils.best_kernel

    def set_obj(self):
        idx = self.obj.row_results.index.values
        descs = self.obj.mini_domain.loc[idx, :]

        target_vals = self.obj.row_results.loc[:, self.obj.target]
        results = pd.concat([descs, target_vals], axis=1)

        self.obj = Objective(results=results,
                             row_results=self.obj.row_results,
                             domain=self.obj.domain,
                             mini_domain=self.obj.mini_domain,
                             all_name_combs=self.all_name_combs,
                             target=self.target,
                             target_scaling=self.target_scaling,
                             opt_type=self.opt_type,
                             gpu=self.gpu)
