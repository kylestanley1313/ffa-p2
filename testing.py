import argparse
import time
import torch
from torch import Generator

import factor_model as fm


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--to_test')
    args = parser.parse_args()


    if args.to_test == 'cuda':

        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

    elif args.to_test == 'ffa':

        # Globals
        seed = 12345
        gen = Generator().manual_seed(seed)

        # Generate factor model data
        loadings = torch.tensor([
            [3, 0.1],
            [-2, 0.3],
            [0.1, 2.5], 
            [-0.2, 4],
            [-0.3, -3]
        ])

        err_sds = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.3])

        print("----- Testing FactorModelDataScratch -----")
        data = fm.FactorModelDataScratch(loadings.T, err_sds, gen=gen, num_train=200, num_val=100)
        batch = next(iter(data.train_dataloader()))
        print(batch)
        print("----- Testing FactorModelDataConcise -----")
        data = fm.FactorModelDataConcise(loadings.T, err_sds, gen=gen, num_train=200, num_val=100)
        batch = next(iter(data.train_dataloader()))
        print(batch)

        # Generate covariance data
        cov_mask = torch.zeros(data.num_vars, data.num_vars)
        for i in range(data.num_vars):
            for j in range(data.num_vars):
                if i <= j:
                    cov_mask[i,j] = 1
        print("----- Testing FeatureCovarianceDataScratch -----")
        cov_data = fm.FeatureCovarianceDataScratch(data, cov_mask, batch_size=3, gen=gen)
        for batch in cov_data.train_dataloader():
            idx, cov = batch
            print(idx, cov)
        print("----- Testing FeatureCovarianceDataConcise -----")
        cov_data = fm.FeatureCovarianceDataConcise(data, cov_mask, batch_size=3, gen=gen)
        for batch in cov_data.train_dataloader():
            idx, cov = batch
            print(idx, cov)

        # Train low rank covariance model
        print("----- Testing LowRankCovarianceModelScratch -----")
        model = fm.LowRankCovarianceModelScratch(data.num_vars, data.num_facs, lr=0.05, gen=gen)
        trainer = fm.Trainer(max_epochs=1000)
        start = time.time()
        trainer.fit(model, cov_data)
        end = time.time()
        print(f"Elapsed Time: {end - start}")
        print("----- Testing LowRankCovarianceModelConcise -----")
        model = fm.LowRankCovarianceModelConcise(data.num_vars, data.num_facs, lr=0.05)
        trainer = fm.Trainer(max_epochs=1000)
        start = time.time()
        trainer.fit(model, cov_data)
        end = time.time()
        print(f"Elapsed Time: {end - start}")

        # NOTE: ModelScratch is slightly faster than ModelConcise

    else: 
        print(f"to_test = {args.to_test} not supported!")