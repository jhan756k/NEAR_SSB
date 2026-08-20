from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from data_prep.data_prep import prepare
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, chirp, stft
import yaml
from main_model import DDPM
from denoising_model_small import ConditionalModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from utils import train, evaluate
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from scipy import stats

import metrics


if __name__ == "__main__": 
    shot_sg = [1,3,5,10]
    device = 'cuda:0'
    
    nts = [1,2]
    
    
    for shots in shot_sg:
        
        ssd_total = []
        mad_total = []
        prd_total = []
        cos_sim_total = []
        snr_noise = []
        snr_recon = []
        snr_improvement = []
        n_level = []
        
        for n_type in nts:
            path = "config/base.yaml"

            current_dir = Path(__file__).resolve().parent
            config_path = current_dir / path

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)    
            
            base_model = ConditionalModel(config['train']['feats']).to('cuda:0')
            model = DDPM(base_model, config, 'cuda:0')
            foldername = current_dir / "check_points" / f"noise_type_{n_type}"
            output_path = foldername / "model.pth"
            
            model.load_state_dict(torch.load(output_path))
            
            
            [_, _, X_test, y_test] = prepare(noise_version=n_type, qt_path="dataset/qtdb", nstdb_path="dataset/mitnoise", output_dir="data_prep")
            
            X_test = torch.FloatTensor(X_test)
            X_test = X_test.permute(0,2,1)
            
            y_test = torch.FloatTensor(y_test)
            y_test = y_test.permute(0,2,1)
            X_test = X_test.detach()
            #test_set = TensorDataset(y_test, X_test)
            test_set = TensorDataset(y_test, X_test)
            
            test_loader = DataLoader(test_set, batch_size=50, num_workers=0)

            n_level.append(np.load(current_dir / 'rnd_test.npy'))
            
            with tqdm(test_loader) as it:
                for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                    clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                
                    if shots > 1:
                        output = 0
                        for i in range(shots):
                            output+=model.denoising(noisy_batch)
                        output /= shots
                    else:
                        output = model.denoising(noisy_batch)
                
                    clean_batch = clean_batch.permute(0, 2, 1)
                    noisy_batch = noisy_batch.permute(0, 2, 1)
                    output = output.permute(0, 2, 1) #B,L,1
                    out_numpy = output.cpu().detach().numpy()
                    clean_numpy = clean_batch.cpu().detach().numpy()
                    noisy_numpy = noisy_batch.cpu().detach().numpy()
                    
                    ssd_total.append(metrics.SSD(clean_numpy, out_numpy))
                    mad_total.append(metrics.MAD(clean_numpy, out_numpy))
                    prd_total.append(metrics.PRD(clean_numpy, out_numpy))
                    cos_sim_total.append(metrics.COS_SIM(clean_numpy, out_numpy))
                    snr_noise.append(metrics.SNR(clean_numpy, noisy_numpy))
                    snr_recon.append(metrics.SNR(clean_numpy, out_numpy))
                    snr_improvement.append(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
                    
                    
        ssd_total = np.concatenate(ssd_total, axis=0)
        mad_total = np.concatenate(mad_total, axis=0)
        prd_total = np.concatenate(prd_total, axis=0)
        cos_sim_total = np.concatenate(cos_sim_total, axis=0)
        snr_noise = np.concatenate(snr_noise, axis=0)
        snr_recon = np.concatenate(snr_recon, axis=0)
        snr_improvement = np.concatenate(snr_improvement, axis=0)
        n_level = np.concatenate(n_level, axis=0)
        
        
        segs = [0.2, 0.6, 1.0, 1.5, 2.0]
        print('===================='+str(shots)+'-shots'+'====================')
        print('====================ALL====================')
        print("ssd: ",ssd_total.mean(),'+/-',ssd_total.std(),)
        print("mad: ", mad_total.mean(),'+/-',mad_total.std(),)
        print("prd: ", prd_total.mean(),'+/-',prd_total.std(),)
        print("cos_sim: ", cos_sim_total.mean(),'+/-',cos_sim_total.std(),)
        print("snr_in: ", snr_noise.mean(),'+/-',snr_noise.std(),)
        print("snr_out: ", snr_recon.mean(),'+/-',snr_recon.std(),)
        print("snr_improve: ", snr_improvement.mean(),'+/-',snr_improvement.std(),)
        
        for idx_seg in range(len(segs) - 1):
            idx = np.argwhere(np.logical_and(n_level>=segs[idx_seg], n_level<=segs[idx_seg+1]))
            
            print('===================='+str(segs[idx_seg]) +'< noise <'+ str(segs[idx_seg+1])+ '====================')
            print("ssd: ",ssd_total[idx].mean(),'+/-',ssd_total[idx].std(),)
            print("mad: ", mad_total[idx].mean(),'+/-',mad_total[idx].std(),)
            print("prd: ", prd_total[idx].mean(),'+/-',prd_total[idx].std(),)
            print("cos_sim: ", cos_sim_total[idx].mean(),'+/-',cos_sim_total[idx].std(),)
            print("snr_in: ", snr_noise[idx].mean(),'+/-',snr_noise[idx].std(),)
            print("snr_out: ", snr_recon[idx].mean(),'+/-',snr_recon[idx].std(),)
            print("snr_improve: ", snr_improvement[idx].mean(),'+/-',snr_improvement[idx].std(),)
            

        

