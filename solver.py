# Some code based on https://github.com/thuml/Anomaly-Transformer



import os

import time

import math

import logging

import builtins

import random

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

from tqdm import tqdm

from utils.utils import *

from model.lr import PolynomialDecayLR

from model.Transformer import TransformerVar

from model.loss_functions import *

from data_factory.data_loader import get_loader_segment

from metrics.metrics import combine_all_evaluation_scores

from sklearn.metrics import (precision_score,

                             recall_score,

                             f1_score,

                             auc,

                             roc_auc_score,

                             average_precision_score,

                             precision_recall_curve,

                             )



from metrics.metrics import *

from metrics import point_adjustment

from metrics import ts_metrics_enhanced



os.environ["CUDA_VISIBLE_DEVICES"] = '0'





def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=2, decay_factor=0.9):

    lr_adjust = {epoch: initial_lr * (decay_factor ** ((epoch - 1) // step_size))}

    if epoch in lr_adjust.keys():

        lr = lr_adjust[epoch]

        for param_group in optimizer.param_groups:

            param_group['lr'] = lr

        print(f'Updating learning rate to {lr}')





class OneEarlyStopping:

    def __init__(self, patience=10, verbose=False, dataset_name='', delta=0, model_version='baseline'):

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta

        self.dataset = dataset_name

        self.model_version = model_version



    def __call__(self, val_loss, model, path):

        score = -val_loss

        if self.best_score is None:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

        elif score < self.best_score + self.delta:

            self.counter += 1

            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.save_checkpoint(val_loss, model, path)

            self.counter = 0



    def save_checkpoint(self, val_loss, model, path):

        if self.verbose:

            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')



        # 根据模型版本生成不同的checkpoint文件名

        model_version = getattr(self, 'model_version', 'baseline')

        checkpoint_name = f'{self.dataset}_{model_version}_checkpoint.pth'

        torch.save(model.state_dict(), os.path.join(path, checkpoint_name))

        self.val_loss_min = val_loss





class Solver(object):

    DEFAULTS = {}



    def __init__(self, config):



        self.scheduler = None

        self.model = None

        self.optimizer = None

        self.__dict__.update(Solver.DEFAULTS, **config)



        # Set random seed for reproducibility
        random_seed = config.get('random_seed', 42)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



        self.train_time_per_epoch = 0.0

        self.test_time_per_epoch = 0.0

        self.model_version = config.get('model_version', 'baseline')

        self.train_loader, self.vali_loader = get_loader_segment(self.data_path,

                                                                 batch_size=self.batch_size,

                                                                 win_size=self.win_size,

                                                                 mode='train',

                                                                 dataset=self.dataset)



        self.test_loader = get_loader_segment(self.data_path,

                                              batch_size=self.batch_size,

                                              win_size=self.win_size,

                                              mode='test',

                                              dataset=self.dataset)



        self.entropy_loss = EntropyLoss()

        self.criterion = nn.MSELoss(reduction='none')



        self.use_contrastive = config.get('use_contrastive', 'False')



        self.use_contrastive = self.use_contrastive if isinstance(self.use_contrastive, bool) else (self.use_contrastive == 'True')



        self.cl_warmup_epochs = config.get('cl_warmup_epochs', 3)



        self.cl_patience = config.get('cl_patience', 2)



        self.cl_min_delta = config.get('cl_min_delta', 1e-4)



        cl_reenable = config.get('cl_reenable', 'True')



        self.cl_reenable = cl_reenable if isinstance(cl_reenable, bool) else (cl_reenable == 'True')



        self.best_val_cl_loss = np.Inf



        self.cl_bad_epochs = 0



        self.cl_curr_enabled = self.use_contrastive

        

        # ε-tube 损失（借鉴 SVR 思想）

        use_eps = config.get('use_epsilon_tube', False)

        self.use_epsilon_tube = use_eps if isinstance(use_eps, bool) else (use_eps == 'True')

        self.epsilon_init = config.get('epsilon_init', 0.1)

        self.adaptive_epsilon_mode = config.get('adaptive_epsilon_mode', 'none')

        if self.use_epsilon_tube:

            from model.loss_functions import SmoothEpsilonTubeLoss

            self.epsilon_tube_loss = SmoothEpsilonTubeLoss(

                epsilon=self.epsilon_init, 

                learnable=True,

                adaptive_mode=self.adaptive_epsilon_mode

            )

        

        # ProtoSVDD 损失（基于原型库的SVDD）

        use_svdd = config.get('use_svdd', False)

        self.use_svdd = use_svdd if isinstance(use_svdd, bool) else (use_svdd == 'True')

        self.lambda_svdd = config.get('lambda_svdd', 1.0)  # 默认权重改为1.0
        self.lambda_consist = config.get('lambda_consist', 0.2)

        if self.use_svdd:

            from model.loss_functions import ProtoSVDDLoss

            self.proto_svdd_loss = ProtoSVDDLoss(

                nu=0.05,

                R=2.0,

                lambda_consist=self.lambda_consist,

                temperature=0.15

            )



        self.logger = logging.getLogger()

        self.logger.setLevel(logging.INFO)



        formatter = logging.Formatter('%(asctime)s - %(message)s')

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(formatter)



        # Check if the stream handler is already added

        if not any(isinstance(handler, logging.StreamHandler) for handler in self.logger.handlers):

            self.logger.addHandler(stream_handler)

            # Redirect print to logger

            self._redirect_print_to_logger()



    def _redirect_print_to_logger(self):

        def print_to_logger(*args, **kwargs):

            message = " ".join(map(str, args))

            self.logger.info(message)

            # Replace the built-in print function with the custom one

            builtins.print = print_to_logger



    def model_init(self, config):

        self.model = TransformerVar(config)



        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),

                                           lr=self.peak_lr, weight_decay=self.weight_decay)

        self.scheduler = PolynomialDecayLR(self.optimizer,

                                           warmup_updates=self.warmup_epoch * self.batch_size,

                                           tot_updates=self.num_epochs * self.batch_size,

                                           lr=self.peak_lr,

                                           end_lr=self.end_lr,

                                           power=1.0)



        if torch.cuda.is_available():

            self.model = torch.nn.DataParallel(self.model, device_ids=[0], output_device=0).to(self.device)

        

        # Move epsilon_tube_loss to device

        if self.use_epsilon_tube:

            self.epsilon_tube_loss = self.epsilon_tube_loss.to(self.device)

        

        # Move proto_svdd_loss to device

        if self.use_svdd:

            self.proto_svdd_loss = self.proto_svdd_loss.to(self.device)



    def _compute_anomaly_score(self, rec_loss, output_dict, gathering_loss_fn):

        """

        统一的异常评分计算：组合记忆库和SVDD评分

        

        Args:

            rec_loss: (B, T, C) 重构误差

            output_dict: 模型输出字典

            gathering_loss_fn: GatheringLoss 实例

        

        Returns:

            loss: (B, T) 异常分数

        """

        base_score = rec_loss.mean(dim=-1)  # (B, T)

        

        # 记忆库评分

        queries = output_dict['queries']

        mem_items = output_dict['mem']

        if mem_items is not None:

            latent_score = torch.softmax(gathering_loss_fn(queries, mem_items) / self.temperature, dim=-1)

            memory_score = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)

        else:

            memory_score = base_score

        

        # ProtoSVDD 评分

        if self.use_svdd:

            z_svdd_time = output_dict.get('z_svdd_time', None)

            z_svdd_freq = output_dict.get('z_svdd_freq', None)

            

            if z_svdd_time is not None and z_svdd_freq is not None:

                # 获取模型的原型库 mem_R

                mem_R = self.model.module.mem_R if hasattr(self.model, 'module') else self.model.mem_R

                svdd_score = self.proto_svdd_loss.anomaly_score(z_svdd_time, z_svdd_freq, mem_R)

            else:

                svdd_score = torch.zeros_like(base_score)

        else:

            svdd_score = torch.zeros_like(base_score)

        

        # 组合评分

        if mem_items is not None and self.use_svdd:

            # 两者都启用：记忆库为主 + SVDD 辅助

            loss = memory_score + 0.3 * svdd_score

        elif mem_items is not None:

            # 只有记忆库

            loss = memory_score

        elif self.use_svdd:

            # 只有 SVDD

            loss = base_score + svdd_score

        else:

            # 都没有

            loss = base_score

        

        return loss



    def vali(self, vali_loader):

        self.model.eval()



        valid_loss_list = []

        valid_re_loss_list = []

        valid_intra_loss_list = []

        valid_cl_loss_list = []



        for i, (input_data, _) in enumerate(vali_loader):

            input_data = input_data.float().to(self.device)

            output_dict = self.model(input_data, mode='vali_cl')



            output = output_dict['out']

            attn = output_dict['attn']



            cl_loss = output_dict.get('cl_loss', torch.tensor(0.0).to(self.device))



            rec_loss = self.criterion(output, input_data).mean()

            attn_loss = torch.zeros_like(rec_loss) if attn is None else self.entropy_loss(attn) * self.alpha



            loss = rec_loss + attn_loss



            valid_re_loss_list.append(rec_loss.detach().cpu().numpy())

            valid_intra_loss_list.append(attn_loss.detach().cpu().numpy())

            valid_cl_loss_list.append(cl_loss.detach().cpu().numpy())

            valid_loss_list.append(loss.detach().cpu().numpy())



        return np.average(valid_loss_list), np.average(valid_re_loss_list), np.average(valid_intra_loss_list), np.average(valid_cl_loss_list)



    def train(self):



        # print("======================TRAIN MODE======================")

        if not os.path.exists(self.model_save_path):

            os.makedirs(self.model_save_path)

        early_stopping = OneEarlyStopping(patience=self.patience, verbose=True, dataset_name=self.dataset, model_version=self.model_version)

        train_steps = len(self.train_loader)



        training_start_time = time_now = time.time()



        for epoch in tqdm(range(self.num_epochs)):



            if self.use_contrastive:



                model_ref = self.model.module if hasattr(self.model, 'module') else self.model



                model_ref.set_contrastive_runtime_state(self.cl_curr_enabled)

            iter_count = 0

            loss_list = []

            rec_loss_list = []

            intra_loss_list = []



            # adjust_learning_rate(self.optimizer, epoch, self.peak_lr)

            epoch_time = time.time()



            self.model.train()

            for i, (input_data, labels) in enumerate(self.train_loader):



                self.optimizer.zero_grad()

                iter_count += 1

                input_data = input_data.float().to(self.device)

                output_dict = self.model(input_data, mode='train')



                output = output_dict['out']

                out_t = output_dict.get('out_t', None)

                out_f = output_dict.get('out_f', None)

                attn = output_dict['attn']

                cl_loss = output_dict.get('cl_loss', torch.tensor(0.0).to(self.device))

                z_svdd_time = output_dict.get('z_svdd_time', None)

                z_svdd_freq = output_dict.get('z_svdd_freq', None)



                # 选择重构损失：ε-tube 或 MSE

                if self.use_epsilon_tube:

                    rec_loss = self.epsilon_tube_loss(output, input_data)

                    # 双分支对称重构：额外计算各分支的重构损失

                    if out_t is not None and out_f is not None:

                        rec_loss_t = self.epsilon_tube_loss(out_t, input_data)

                        rec_loss_f = self.epsilon_tube_loss(out_f, input_data)

                        rec_loss = rec_loss + 0.5 * (rec_loss_t + rec_loss_f)

                else:

                    rec_loss = self.criterion(output, input_data).mean()

                    # 双分支对称重构：额外计算各分支的重构损失

                    if out_t is not None and out_f is not None:

                        rec_loss_t = self.criterion(out_t, input_data).mean()

                        rec_loss_f = self.criterion(out_f, input_data).mean()

                        rec_loss = rec_loss + 0.5 * (rec_loss_t + rec_loss_f)

                attn_loss = torch.zeros_like(rec_loss) if attn is None else self.entropy_loss(attn) * self.alpha



                # ProtoSVDD 损失

                svdd_loss = torch.tensor(0.0).to(self.device)

                if self.use_svdd and z_svdd_time is not None and z_svdd_freq is not None:

                    # 获取模型的原型库 mem_R

                    mem_R = self.model.module.mem_R if hasattr(self.model, 'module') else self.model.mem_R

                    # 前10个epoch动态更新半径

                    update_R = (epoch < 10)

                    svdd_loss, svdd_metrics = self.proto_svdd_loss(z_svdd_time, z_svdd_freq, mem_R, update_R=update_R)

                    svdd_loss = self.lambda_svdd * svdd_loss

                    

                    if i % 100 == 0:

                        print(f"ProtoSVDD: R={svdd_metrics['R']:.3f}, assign={svdd_metrics['assign']:.3f}")



                loss = rec_loss + attn_loss + cl_loss + svdd_loss



                loss_list.append(loss.detach().cpu().numpy())

                rec_loss_list.append(rec_loss.detach().cpu().numpy())

                intra_loss_list.append(attn_loss.detach().cpu().numpy())



                if (i + 1) % 100 == 0:

                    speed = (time.time() - time_now) / iter_count

                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)

                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))

                    iter_count = 0

                    time_now = time.time()

                loss.backward()

                self.optimizer.step()



            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))



            train_loss = np.average(loss_list)

            train_rec_loss = np.average(rec_loss_list)

            train_intra_loss = np.average(intra_loss_list)

            valid_loss, valid_re_loss, valid_intra_loss, valid_cl_loss = self.vali(self.vali_loader)



            if self.use_contrastive:



                if epoch + 1 >= self.cl_warmup_epochs:



                    if valid_cl_loss < self.best_val_cl_loss - self.cl_min_delta:



                        self.best_val_cl_loss = valid_cl_loss



                        self.cl_bad_epochs = 0



                        self.cl_curr_enabled = True



                    else:



                        self.cl_bad_epochs += 1



                        if self.cl_bad_epochs >= self.cl_patience:



                            self.cl_curr_enabled = False



                        elif self.cl_reenable and self.best_val_cl_loss < np.Inf:



                            self.cl_curr_enabled = True



                else:



                    self.best_val_cl_loss = min(self.best_val_cl_loss, valid_cl_loss)



            print(

                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | Vali Loss: {valid_loss:.7f}")



            print(

                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train reconstruction Loss: {train_rec_loss:.7f} | Entropy Loss : {train_intra_loss:.7f}")



            print(

                f"Epoch: {epoch + 1}, Steps: {train_steps} | Valid reconstruction Loss: {valid_re_loss:.7f} | Entropy Loss : {valid_intra_loss:.7f}")



            if self.use_contrastive:



                print(



                    f"Epoch: {epoch + 1}, Steps: {train_steps} | Valid contrastive Loss: {valid_cl_loss:.7f} | Contrastive Enabled Next Epoch: {self.cl_curr_enabled}")



            early_stopping(valid_loss, self.model, self.model_save_path)

            if early_stopping.early_stop:

                print("Early stopping")

                break



        elapsed = time.time() - training_start_time
        self.train_time_per_epoch = round(elapsed / max(self.num_epochs, 1), 3)



        return

    def test(self, params):
        # 根据模型版本加载对应的checkpoint文件
        model_version = getattr(self, 'model_version', 'baseline')
        checkpoint_name = f'{self.dataset}_{model_version}_checkpoint.pth'
        model_name = os.path.join(str(self.model_save_path), checkpoint_name)
        
        try:
            self.model.load_state_dict(torch.load(model_name))
            self.model.eval()
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint {checkpoint_name}: {e}")
            print(f"[INFO] Deleting corrupted checkpoint and skipping this test...")
            if os.path.exists(model_name):
                os.remove(model_name)
            # Return dummy results to allow the experiment to continue
            return {
                'pc_adjust': 0.0,
                'rc_adjust': 0.0,
                'f1_adjust': 0.0,
                'thresh': 0.0,
                'trt': 0.0,
                'tst': 0.0
            }



        print("============================TEST MODE============================")



        criterion = nn.MSELoss(reduction='none')

        gathering_loss = GatheringLoss(reduction='none', memto_framework=True)



        print(f"Dataset: {self.dataset}")



        # Check if using step models

        is_step_model = self.model_version in ['step1', 'step2', 'step3', 'step4']



        if self.threshold_setting == 'preset':

            train_attens_energy = []

            for i, (input_data, labels) in enumerate(self.train_loader):

                input_data = input_data.float().to(self.device)

                output_dict = self.model(input_data, mode='test')



                output = output_dict['out']

                rec_loss = criterion(input_data, output)



                if is_step_model:

                    loss = rec_loss.mean(dim=-1)

                else:

                    loss = self._compute_anomaly_score(rec_loss, output_dict, gathering_loss)



                cri = loss.detach().cpu().numpy()

                train_attens_energy.append(cri)



            train_attens_energy = np.concatenate(train_attens_energy, axis=0).reshape(-1)

            train_energy = np.array(train_attens_energy)



            valid_attens_energy = []

            for i, (input_data, labels) in enumerate(self.vali_loader):

                input_data = input_data.float().to(self.device)

                output_dict = self.model(input_data, mode='test')



                output = output_dict['out']

                rec_loss = criterion(input_data, output)



                if is_step_model:

                    loss = rec_loss.mean(dim=-1)

                else:

                    loss = self._compute_anomaly_score(rec_loss, output_dict, gathering_loss)



                cri = loss.detach().cpu().numpy()

                valid_attens_energy.append(cri)



            valid_attens_energy = np.concatenate(valid_attens_energy, axis=0).reshape(-1)

            valid_energy = np.array(valid_attens_energy)



            combined_energy = np.concatenate([train_energy, valid_energy], axis=0)



            thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)

            print("Threshold :", thresh)



        test_window_labels = []

        test_window_energy = []



        test_labels = []
        test_attens_energy = []  # 完整分数 (用于 PRF)
        test_simple_energy = []  # 简化分数 (用于 AUC/VUS)

        start_time = time.time()

        # Check if using step models (simplified scoring)
        is_step_model = hasattr(self, 'model_version') and self.model_version in ['step1', 'step2', 'step3', 'step4']

        for i, (input_data, labels) in enumerate(self.test_loader):
            input_data = input_data.float().to(self.device)
            output_dict = self.model(input_data, mode='test')

            output = output_dict['out']
            rec_loss = criterion(input_data, output)

            if is_step_model:
                # For step models: use reconstruction error directly
                loss = rec_loss.mean(dim=-1)  # (B, T)
                simple_loss = loss
            else:
                # 完整分数：包含 ProtoSVDD (用于 PRF)
                loss = self._compute_anomaly_score(rec_loss, output_dict, gathering_loss)
                
                # 简化分数：只用重构+记忆 (用于 AUC/VUS，与您的版本一致)
                queries = output_dict.get('queries', None)
                mem_items = output_dict.get('mem', None)
                if queries is not None and mem_items is not None:
                    latent_score = torch.softmax(gathering_loss(queries, mem_items) / self.temperature, dim=-1)
                    simple_loss = harmonic_loss_compute(rec_loss, latent_score, self.aggregation)
                else:
                    simple_loss = rec_loss.mean(dim=-1)

            cri = loss.detach().cpu().numpy()
            simple_cri = simple_loss.detach().cpu().numpy()
            test_attens_energy.append(cri)
            test_simple_energy.append(simple_cri)
            test_labels.append(labels)



            test_window_energy.extend(cri.mean(axis=-1))

            test_window_labels.extend((labels.sum(axis=-1) > 1).numpy().astype(int))



        self.test_time_per_epoch = round(time.time() - start_time, 3)

        test_attens_energy = np.concatenate(test_attens_energy, axis=0).reshape(-1)
        test_simple_energy = np.concatenate(test_simple_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(test_attens_energy)  # 完整分数 (用于 PRF)
        test_energy_simple = np.array(test_simple_energy)  # 简化分数 (用于 AUC/VUS)
        test_labels = np.array(test_labels)



        if self.threshold_setting == 'optimal':

            anomaly_ratio = self.anomaly_ratio

            thresh = np.percentile(test_energy, 100 - anomaly_ratio)

            print("Threshold :", thresh)

            pred = (test_energy > thresh).astype(int)

            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)

        else:

            results = {k: 0.0 for k in metric_list}



            results['thresh'] = 0.0



            pred = (test_energy > thresh).astype(int)



            gt = test_labels.astype(int)



            print(f"pred: {pred.shape}, gt: {gt.shape}")



            events = get_events(gt)



            _, _, _, _, _, _, threshold_setting_results = get_point_adjust_scores(gt, pred, test_energy, events)



            results = ts_metrics_enhanced(test_labels, point_adjustment(test_labels, test_energy), pred)



            results['pc_adjust'] = threshold_setting_results['pc_adjust']

            results['rc_adjust'] = threshold_setting_results['rc_adjust']

            results['f1_adjust'] = threshold_setting_results['f1_adjust']



        precision_adjust, recall_adjust, f_score_adjust = results['pc_adjust'], results['rc_adjust'], results['f1_adjust']

        results['thresh'] = thresh

        results['trt'] = self.train_time_per_epoch

        results['tst'] = self.test_time_per_epoch



        print('=' * 63)
        print(f"Dataset: {self.dataset} | Precision_adjusted: {precision_adjust:.4f} | Recall_adjusted: {recall_adjust:.4f} | f1_score_adjusted: {f_score_adjust:.4f} ")

        # Standard AUC-ROC / AUC-PR from continuous anomaly scores
        # 使用简化分数 (只用重构+记忆，与您的版本一致)
        try:
            gt = test_labels.astype(int)
            std_auc_roc = roc_auc_score(gt, test_energy_simple)  # 使用简化分数
            std_auc_pr = average_precision_score(gt, test_energy_simple)  # 使用简化分数
            results['auc_roc'] = std_auc_roc
            results['auc_pr'] = std_auc_pr
        except Exception as e:
            pass  # Silently skip

        # Affiliation metrics only
        # 使用简化分数计算 (与您的版本一致)
        try:
            gt = test_labels.astype(int)
            # 使用简化分数重新计算 pred
            pred_simple = (test_energy_simple > thresh).astype(int)
            vus_scores = combine_all_evaluation_scores(pred_simple.copy(), gt.copy())
            results.update(vus_scores)
            # 只打印 Affiliation
            print(f"Dataset: {self.dataset} | Aff_Precision: {vus_scores['Affiliation precision']:.4f} | Aff_Recall: {vus_scores['Affiliation recall']:.4f} | Aff_F1: {vus_scores['Affiliation f1 score']:.4f}")
        except Exception as e:
            print(f"[WARN] Affiliation computation failed: {e}")

        return results









def get_point_adjust_scores(y_test, pred_labels, pred_scores, true_events):

    results = {

        "pc": 0.0,

        "rc": 0.0,

        "f1": 0.0,

        "acc_adjust": 0.0,

        "pc_adjust": 0.0,

        "rc_adjust": 0.0,

        "f1_adjust": 0.0,

        "mcc_adjust": 0.0,

        "prc": 0.0,

        "roc": 0.0,

        "apc": 0.0,

    }

    tp = 0

    fn = 0

    for true_event in true_events.keys():

        true_start, true_end = true_events[true_event]

        if pred_labels[true_start:true_end].sum() > 0:

            tp += (true_end - true_start)

        else:

            fn += (true_end - true_start)

    fp = np.sum(pred_labels) - np.sum(pred_labels * y_test)



    pc, rc, fscore = get_prec_rec_fscore(tp, fp, fn)



    tn = len(pred_labels) - (tp + fp + fn)



    avg_precision = average_precision_score(y_test, pred_scores)

    auc_roc = roc_auc_score(y_test, pred_scores)

    precision, recall, _ = precision_recall_curve(y_true=y_test, y_score=pred_scores)



    results['pc'] = round(precision_score(y_test, pred_labels, average='binary'), 4)

    results['rc'] = round(recall_score(y_test, pred_labels, average='binary'), 4)

    results['f1'] = round(f1_score(y_test, pred_labels, average='binary'), 4)



    results['f1_adjust'] = round(fscore, 4)

    results['pc_adjust'] = round(pc, 4)

    results['rc_adjust'] = round(rc, 4)

    # results['mcc_adjust'] = round(matthews_correlation_coefficient(tp, tn, fp, fn), 4)

    results['acc_adjust'] = round((tp + tn) / len(y_test), 4)



    results["prc"] = round(auc(recall, precision), 4)

    results["roc"] = round(auc_roc, 4)

    results["apc"] = round(avg_precision, 4)



    return fp, fn, tp, pc, rc, fscore, results





def matthews_correlation_coefficient(TP, TN, FP, FN):

    numerator = TP * TN - FP * FN

    denominator = np.sqrt(TP + FP) * np.sqrt(TP + FN) * np.sqrt(TN + FP) * np.sqrt((TN + FN))



    # Avoid division by zero

    if denominator < np.finfo(float).eps:

        return 0.0



    mcc = numerator / denominator

    return mcc





def get_f_score(pc, rc):

    if pc == 0 and rc == 0:

        f_score = 0

    else:

        f_score = 2 * (pc * rc) / (pc + rc)

    return f_score





def get_prec_rec_fscore(tp, fp, fn):

    if tp == 0:

        precision = 0

        recall = 0

    else:

        precision = tp / (tp + fp)

        recall = tp / (tp + fn)

    fscore = get_f_score(precision, recall)

    return precision, recall, fscore





def get_events(y_test, outlier=1, normal=0, breaks=[]):

    events = dict()

    label_prev = normal

    event = 0  # corresponds to no event

    event_start = 0

    for tim, label in enumerate(y_test):

        if label == outlier:

            if label_prev == normal:

                event += 1

                event_start = tim

            elif tim in breaks:

                # A break point was hit, end current event and start new one

                event_end = tim - 1

                events[event] = (event_start, event_end)

                event += 1

                event_start = tim

        else:

            # event_by_time_true[tim] = 0

            if label_prev == outlier:

                event_end = tim - 1

                events[event] = (event_start, event_end)

        label_prev = label



    if label_prev == outlier:

        # event_end = tim - 1  # original code is wrong!

        event_end = tim

        events[event] = (event_start, event_end)

    return events

