import torch
from datasets.main import load_dataset
from model import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate
import random
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image
import models_vit
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
#from defaults import assert_and_infer_cfg, get_cfg
#from models.build import build_model
#sys.path.append("..")
#from segment_anything import sam_model_registry, SamPredictor

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    # cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def deactivate_batchnorm(m):
    '''
        Deactivate batch normalisation layers
    '''
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, alpha, anchor, device, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.v = v
        self.alpha = alpha
        self.anchor = anchor
        self.device = device

    def forward(self, output1, vectors, label):
        '''
        Args:
            output1 - feature embedding/representation of current training instance
            vectors - list of feature embeddings/representations of training instances to contrast with output1
            label - value of zero if output1 and all vectors are normal, one if vectors are anomalies
        '''

        euclidean_distance = torch.FloatTensor([0]).to(self.device)

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)/ torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device))

        # 在总距离上加上了Loss_stop:output1和anchor之间的距离
        euclidean_distance += self.alpha*((F.pairwise_distance(output1, self.anchor)) /torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device) )

        #calculate the margin
        marg = (len(vectors) + self.alpha) * self.margin

        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss

        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        return loss_contrastive



def create_batches(lst, n):
    '''
    Args:
        lst - list of indexes for training instances
        n - batch size
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def train(model, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, smart_samp, k, eval_epoch, model_type, bs, num_ref_eval, num_ref_dist, device):

    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    patience = 0
    max_patience = 2 #patience based on train loss 
    max_iter = 0
    patience2 = 99 #patience based on evaluation AUC 初始值为10
    best_val_auc = 0
    best_val_auc_min = 0
    best_f1=0
    best_acc=0
    stop_training = False
    classes = ['bottle' , 'cable',  'capsule',  'carpet',  'grid',  'hazelnut', 'leather', 'metal_nut',  'pill',  'screw',  'tile',  'toothbrush',  'transistor',  'wood' , 'zipper']
    class_name = classes[normal_class]
    start_time = time.time()

    for epoch in range(epochs):
        print("Starting epoch " + str(epoch+1))
        if epoch == 30 :
            print("test")

        model.train()

        loss_sum = 0

        #mask = useSAM(img1, predictor) # type:numpy
        #mask2 = useSAM(img2, predictor)
        mask_path = 'masks/{}_2.pt'.format(class_name)
        mask = torch.load(mask_path)
                # mask = torch.load('/home/traffic/Documents/ACode/FewSOME/src/masks/capsule_2.pt')
        mask = np.logical_not(mask)
        mask_tensor = torch.from_numpy(mask) # type:tensor
        mask_bir = mask_tensor.type(torch.float32)
        mask_new_resized = mask_bir.view(1,1,64,64).to("cuda")
                # 将mask_new的形状调整为与attn_weg相同的形状
                # mask_new_resized = F.interpolate(mask_new, size=(65, 65), mode='nearest')
        count = torch.sum(mask_new_resized == 1)    # 2.5k
                # mask_new_resized[mask_new_resized == 0] += 0.3 # 相当于给非目标区域（０区域即背景）减少权重，即更关注目标区域
                # mask_new_resized = mask_new_resized.to("cuda")
        num_masks = 12  # 定义要生成的 mask 数量
        delta = 0.1  # 定义每次递增的值
        start = 0.9
        masks = []  # 存储生成的各个 mask

        for i in range(num_masks):
            new_value = start - (i * delta)  # 计算当前 mask 的增量值
            if new_value < 0 :
                new_value = 0
            mask_i = mask_new_resized.clone()  # 创建 mask_i 作为 mask_new_resized 的副本
            mask_i[mask_new_resized == 0] = new_value + mask_i[mask_new_resized == 0]  # 在 mask_i 的基础上添加增量值
            mask_i = mask_i.to("cuda")
            masks.append(mask_i)

        #create batches for epoch
        np.random.seed(epoch)
        np.random.shuffle(ind)
        batches = list(create_batches(ind, bs))
#        base_ind = -1 # 自己设置的，即不需要anchor
        #iterate through each batch
        for i in range(int(np.ceil(len(ind) / bs))):
            first_block = model.blocks[0]
            #iterate through each training instance in batch
            for batch_ind,index in enumerate(batches[i]):

                seed = (epoch+1) * (i+1) * (batch_ind+1)
                img1, img2, labels, base = train_dataset.__getitem__(index, seed, base_ind)

                # Forward
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                
                
                # masks = masks.to("cuda")
                
                if (index ==base_ind):
                  output1 = anchor
                else:
                  """
                  for i, block in enumerate(model.blocks):
                      
                      attn_weg = block.attn_weg
                      masks[i] = F.pad(masks[i], (0, 1, 0, 1), value=1)
                      
                      #cls_token = attn_weg[:, :, 0:1, 0:1]
                      #remaining = attn_weg[:, :, 1:, 1:]
                      #flag =  mask_new_resized * remaining
                      #weg_new = torch.cat([cls_token, flag], dim=2)
                      weg_new =  masks[i] * attn_weg
                      # weg_new =  attn_weg * mask_x
                      block.weg_new = weg_new
                      block.weg_new = None
                      block.attn.attn_new = None
                  """
                  
                  attn_weg1 = first_block.attn_weg.clone()
                  value = 0.6
                  mask_new_resized[mask_new_resized == 0] = value + mask_new_resized[mask_new_resized == 0]
                #print(mask_new_resized.shape)
                  if mask_new_resized.size(2) == 64:
                      mask_new_resized = F.pad(mask_new_resized, (0, 1, 0, 1), value=1)
                  weg_new1 =  attn_weg1 * mask_new_resized.clone()
                  first_block.attn.update_weg(weg_new1)
                #first_block.weg_new = weg_new1
                  #first_block.attn.attn_new.detach()
                  first_block.attn.attn_new = None  

                  
                  output1 = model.forward(img1.float())  # output1:(1,1000)
                  first_block.attn.attn_new.detach()
                  first_block.weg_new = None
                  first_block.attn.attn_new = None

                
                

                #atten1 = model.atten
                #model.atten = None
                
                if (smart_samp == 0) & (k>1):
                  first_block.attn.attn_new = None
                  first_block.weg_new = None

                  vecs=[]
                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, k):
                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vecs.append(output2)

                elif smart_samp == 0:
                  first_block.attn.attn_new = None
                  first_block.weg_new = None
                
                  if (base == True):
                    output2 = anchor
                  else:
                    output2 = model.forward(img2.float()) # output2:(1,1000)
                    #first_block.attn.attn_new.detach()
                    first_block.weg_new = None
                    first_block.attn.attn_new = None
                  
                  vecs = [output2]
                


                else:
                  max_eds = [0] * k
                  max_inds = [-1] * k
                  max_ind =-1
                  vectors=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, num_ref_dist):

                    if ((num_ref_dist ==1) & (ind2[j] == base_ind)) | ((num_ref_dist ==1) & (ind2[j] == index)):
                        c = 0
                        while ((ind2[j] == base_ind) | (index == ind2[j])):
                            np.random.seed(seed * c)
                            j = np.random.randint(len(ind) )
                            c = c+1

                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vectors.append(output2)
                      euclidean_distance = F.pairwise_distance(output1, output2)

                      for b, vec in enumerate(max_eds):
                          if euclidean_distance > vec:
                            max_eds.insert(b, euclidean_distance)
                            max_inds.insert(b, len(vectors)-1)
                            if len(max_eds) > k:
                              max_eds.pop()
                              max_inds.pop()
                            break

                  vecs = []

                  for x in max_inds:
                      with torch.no_grad():
                          vecs.append(vectors[x])
                #first_block.attn.attn_new.detach()
                first_block.weg_new = None
                first_block.attn.attn_new = None
                """
                atten2 = model.atten
                if atten1 is not None and atten2 is not None:
                    new_input1 = torch.cat((atten1, atten2), dim=1)
                    new_output1 = model.self_atten(new_input1)
                else:
                    new_output1 = output1.clone()
                model.atten = None
                """


                if batch_ind ==0:
                    loss = criterion(output1,vecs,labels)
                    # loss = criterion(new_output1,vecs,labels)
                else:
                    loss = loss + criterion(output1,vecs,labels)
                    #loss = loss + criterion(new_output1,vecs,labels)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            #loss.backward()
            optimizer.step()

            

            



        train_losses.append((loss_sum / len(ind))) #average loss for each training instance

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))


        if (eval_epoch == 1):
            training_time = time.time() - start_time
            eval_start_time = time.time()
            val_auc, val_loss, val_auc_min, f1, acc,df, ref_vecs, inf_times, total_times = evaluate(anchor, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, data_path, criterion, alpha, num_ref_eval, device)
            eval_time = time.time() - eval_start_time
            print('Validation AUC is {}'.format(val_auc))
            print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))
            if val_auc_min > best_val_auc_min:
                best_val_auc = val_auc
                best_val_auc_min = val_auc_min
                best_epoch = epoch
                best_f1 = f1
                best_acc = acc
                best_df=df
                max_iter = 0
                training_time_best = (time.time() - start_time) - (eval_time*(epoch+1))

                training_time = (time.time() - start_time) - (eval_time*(epoch+1))
                write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time_best,f1,acc, train_losses, inf_times, total_times)

            else:
                max_iter+=1

            if max_iter == patience2:
                break

        elif args.early_stopping ==1:
            if epoch > 1:
              decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)

              if decrease <= 0.5:
                patience += 1


              if (patience==max_patience) | (epoch == epochs-1):
                  stop_training = True


        elif (epoch == (epochs -1)) & (eval_epoch == 0):
            stop_training = True




        if stop_training == True:
            print("--- %s seconds ---" % (time.time() - start_time))
            training_time = time.time() - start_time
            val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs, inf_times, total_times = evaluate(anchor,seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, data_path, criterion, alpha, num_ref_eval, device)


            write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times)

            break



    print("Finished Training")
    if eval_epoch == 1:
        print("AUC was {} on epoch {}".format(best_val_auc_min, best_epoch+1))
        return best_val_auc, best_epoch, best_val_auc_min, training_time_best, best_f1, best_acc,train_losses
    else:
        print("AUC was {} on epoch {}".format(val_auc_min, epoch+1))
        return val_auc, epoch, val_auc_min, training_time, f1,acc, train_losses




def write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times):
    '''
        Write out results to output directories and save model
    '''

    model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
    for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
    torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)


    for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
    df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

    for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
      if (model_name in f) :
        os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
    ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)


    pd.DataFrame([np.mean(inf_times), np.std(inf_times), np.mean(total_times), np.std(total_times), val_auc_min ,f1,acc]).to_csv('./outputs/inference_times/class_'+str(normal_class)+'/'+model_name_temp)

     #write out all details of model training
    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc']
    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, args.alpha, args.lr, args.weight_decay, args.vector_size, args.biases, args.smart_samp, args.k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time,f1,acc]
    string = './outputs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name)
    pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)


def init_feat_vec(model,base_ind, train_dataset, device ):
        '''
        Initialise the anchor
        Args:
            model object
            base_ind - index of training data to convert to the anchor
            train_dataset - train dataset object
            device
        '''

        model.eval()
        anchor,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          anchor = model.forward(anchor.to(device).float())

        return anchor



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    '''
    Get indexes for reference set
    Include anomalies in the reference set if contamination > 0
    Args:
        contamination - level of contamination of anomlies in reference set
        dataset name
        normal class
        task - train/test/validate
        data_path - path to data
        download data
        N - number in reference set
        seed
    '''
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class,task, data_path, download_data) #get all training data
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.ceil(N*contamination)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0] #get indexes of non-normal class
      samp = random.sample(range(0, len(con)), int(numb))
      samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
      final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MVTEC_VGG3','MNIST_VGG3', 'RESNET', 'FASHION_VGG3','ViT','MViT'], required=False)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 1001)
    parser.add_argument('--weight_init_seed', type=int, default = 1001)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 1)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=None)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chp', default=True)
    parser.add_argument('--opts', default=None)
    parser.add_argument('--cfg', dest="cfg_file", help="Path to the config file", default="MVITv2_B.yaml", type=str)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    #cfg = load_config(args)
    #cfg = assert_and_infer_cfg(cfg)
    N = args.num_ref
    num_ref_eval = args.num_ref_eval
    num_ref_dist = args.num_ref_dist
    if num_ref_eval == None:
        num_ref_eval = N
    if num_ref_dist == None:
        num_ref_dist = N

    #if indexes for reference set aren't provided, create the reference set.
    if args.dataset != 'mvtec':
        if args.index != []:
            indexes = [int(item) for item in indexes.split(', ')]
        else:
            indexes = create_reference(args.contamination, args.dataset, args.normal_class, 'train', args.data_path, args.download_data, N, args.seed)



    #create train and test set
    if args.dataset =='mvtec':
        train_dataset = load_dataset(args.dataset, args.index, args.normal_class, 'train',  args.data_path, args.download_data, args.seed, N=N)
        indexes = train_dataset.indexes
    else:
        train_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'train',  args.data_path, download_data = args.download_data)
    if args.task != 'train':
        val_dataset = load_dataset(args.dataset, indexes,  args.normal_class, 'test', args.data_path, download_data=False)
    else:
        val_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'validate', args.data_path, download_data=False)




    #set the seed
    torch.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed_all(args.weight_init_seed)

    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')

    string = './outputs/models/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')

    string = './outputs/ED/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/ref_vec'):
        os.makedirs('outputs/ref_vec')

    string = './outputs/ref_vec/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/losses'):
        os.makedirs('outputs/losses')

    string = './outputs/losses/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/ref_vec_by_pass/'):
        os.makedirs('outputs/ref_vec_by_pass')

    string = './outputs/ref_vec_by_pass/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/inference_times'):
        os.makedirs('outputs/inference_times')
    if not os.path.exists('outputs/inference_times/class_' + str(args.normal_class)):
        os.makedirs('outputs/inference_times/class_'+str(args.normal_class))

    model_n = 'vit_base_patch16'
    # 构建模型
    model = models_vit.__dict__[model_n](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool='',
        img_size = 128
    )

    if args.chp:
        checkpoint_dir = '/content/gdrive/mae_pretrain_vit_base.pth'
        checkpoint = torch.load(checkpoint_dir, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % checkpoint_dir)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                #print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # 插值位置嵌入
        interpolate_pos_embed(model, checkpoint_model)

        # 加载预训练模型
        msg = model.load_state_dict(checkpoint_model, strict=False)
        #print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
    
    else:
        #Initialise the model
        if args.model_type == 'CIFAR_VGG3':
            if args.pretrain == 1:
                model = CIFAR_VGG3_pre(args.vector_size, args.biases)
            else:
                model = CIFAR_VGG3(args.vector_size, args.biases)
        elif args.model_type == 'MNIST_VGG3':
            if args.pretrain == 1:
                model = MNIST_VGG3_pre(args.vector_size, args.biases)
            else:
                model = MNIST_VGG3(args.vector_size, args.biases)
        elif args.model_type == 'RESNET':
            model = RESNET_pre( )
        elif (args.model_type == 'FASHION_VGG3'):
            if (args.pretrain ==1):
                model = FASHION_VGG3_pre(args.vector_size, args.biases)
            else:
                model = FASHION_VGG3(args.vector_size, args.biases)
        elif args.model_type == 'MViT':
            pass
            #model = build_model(cfg)
        """
        elif args.model_type == 'ViT':
            model  = model = VisionTransformer(
                patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
"""
        if (args.model_type == 'RESNET'):
            model.apply(deactivate_batchnorm)




    model.to(args.device)

    model_name = args.model_name + '_normal_class_' + str(args.normal_class) + '_seed_' + str(args.seed)

    #initialise the anchor
    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    np.random.seed(args.epochs)
    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]
    anchor = init_feat_vec(model,base_ind , train_dataset, args.device)


    criterion = ContrastiveLoss(args.alpha, anchor, args.device, args.v)
#    criterion = nn.CrossEntropyLoss()
    auc, epoch, auc_min, training_time, f1,acc, train_losses= train(model,args.lr, args.weight_decay, train_dataset, val_dataset, args.epochs, criterion, args.alpha, model_name, indexes, args.data_path, args.normal_class, args.dataset, args.smart_samp,args.k, args.eval_epoch, args.model_type, args.batch_size, num_ref_eval, num_ref_dist, args.device)
