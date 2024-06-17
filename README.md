# 使用

需要关注的三个项目：

```
git clone git@github.com:ZhuJiaqi9905/varuna.git

git clone git@github.com:NVIDIA/Megatron-LM.git -b varuna Megatron-LM-varuna

git@github.com:ZhuJiaqi9905/apex.git -b varuna apex
```
但是我已经在gpu-46 gpu-90 gpu-91 gpu-92创建好16个容器了。你直接用他们就行。

gpu-91的varu-0是master。
## varuna

安装：

```
python setup.py install
cp varuna/genschedule /opt/conda/lib/python3.6/site-packages/varuna-0.0.1-py3.6.egg/varuna/
```

`start_container.sh`里面是启动容器的脚本。

启动容器后，需要对每个容器修改ssh port并启动ssh server。

## apex

安装
```
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Megatron-LM-varuna

在scripts下有pretrain, proflle, kill的脚本。

在available_machines.out文件中指定node，会在训练时作为跑模型的node。

训练的log在`ssh_logs`文件夹里。

`/mnt/gpu-91`是共享存储。varuna中数据集和checkpoint都需要写在共享存储里面。

varuna只能用gloo后端。

# 一些注意的点

- 如果改了varuna的代码之后，需要运行：
```
python setup.py install
cp varuna/genschedule /opt/conda/lib/python3.6/site-packages/varuna-0.0.1-py3.6.egg/varuna/
```

- 如果gloo连接建不起来，查看一下`/etc/hosts`文件：需要把主机名和ip地址配对。不能用127.0.0.1

- 有一些配置下，模型训练不起来。

- 理论上varuna可以根据profile的结果自动选择最优的切分方案。但是待探索出来怎么用。




# _Varuna_

_Varuna_ is a tool for efficient training of large DNN models on commodity GPUs and networking. It implements a combination of pipeline parallelism and data parallelism in PyTorch, and enables training on a changing set of resources smoothly.

This repository is an implementation of the paper:

["Varuna: Scalable, Low-cost Training of Massive Deep Learning Models"](https://arxiv.org/abs/2111.04007), to appear in EuroSys'22.

## Setup & Installation

Varuna requires python 3, [PyTorch](https://pytorch.org/get-started/locally/) (1.5+) and [apex](https://github.com/NVIDIA/apex). 

The patch `apex.patch` in this directory needs to be applied to apex before building it. Varuna's code and this patch has been tested for [this commit](https://github.com/NVIDIA/apex/commit/0c2c6eea6556b208d1a8711197efc94899e754e1) of apex.
~~~~
git clone https://github.com/NVIDIA/apex
cp apex.patch /path/to/apex/
cd /path/to/apex
git apply apex.patch
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
~~~~
To install, clone this repository, cd into it and run
~~~~
python setup.py install
cp varuna/genschedule /opt/conda/lib/python3.6/site-packages/varuna-0.0.1-py3.6.egg/varuna/
~~~~
## Running

使用GLOO的时候如果发现总是连接127.0.0.1。就要修改`/etc/hosts`文件。

还在`~/.bashrc`中添加了`GLOO_SOCKET_IFNAME`

Varuna trains large DNN models by parallelising them into sequential pipeline stages and data parallel replicas across a set of GPUs. These methods are called pipeline parallelism and data parallelism respectively.
To enable parallel training with Varuna, there are several steps the user must follow. Detailed docs are in the `docs/` folder as webpages (`html/index.html`) or pdf (`varuna.pdf`). 

Examples of models working with varuna can also be found in `examples/`. Please see this folder to run examples with BERT and Megatron-LM.

Some of the steps for are briefly described below.

### CutPoint demarcation

Varuna slices a DNN model into sequential pipeline stages. For this, the model should be annotated with varuna `CutPoint` instances between different operations/ parts of model computation. These are nn.Module instances that are potential slice points in the mode. For each CutPoint, Varuna can either ignore it or activate it as a partition boundary. (see [CutPoints](docs/html/cutpoint.html))

~~~~
from varuna import CutPoint

class SampleModel(nn.Module):
  def __init__(...):
    ....
    self.cutpoints = [CutPoint() for i in range(num_cutpoints)]
    ....

  def forward(input...):
    input = self.some_operation(input)
    input = self.cutpoints[0](input)     # marked as a potential stage boundary
    input = self.some_other_operation(input)
    ....
    for i in range(sub_modules):
      x = sub_module_i(input, ...)
      x = self.cutpoints[i+1](x)        # each cutpoint instance should be used only once in a model
    ....

~~~~

Operations separated by CutPoints should preferably have no shared modules/parameters. For weight sharing between different parts of the module, you should register separate nn.Parameter instances (even for the same tensor) and pass the pair of parameter names as shared_weights to Varuna.

### Wrapping the model in Varuna

The nn.Module for your DNN instance should be wrapped in a `Varuna` instance before training and before optimizer creation. (see [Varuna](docs/html/varuna.html)) Wrapping in `Varuna` returns a model partitioned according to the given stage_to_rank_map (which is passed by the varuna launcher) and moved to the GPU. After this initialization, each rank in the job has only the parts of the model required by it. Varuna internally handles fp16 mixed precision training and shared parameters (such as the initial and last embedding weights in BERT/GPT-2). 
Optimizer creation should be after this since it requires model parameters as input. The optimizer needs to be registered with Varuna using a setter.
~~~~
    model = MyModel()             # full model on CPU
    # provide dummy input function to varuna for initialization. Inputs mst be in dictionary form.
    def get_batch_fn(size, device=None):
        batch = dataset[:size]
        if device is not None:
          batch = [t.to(device) for t in batch]
        inputs, mask = batch
        return {'inputs': inputs, 'mask': mask, 'extra_norm': True }

    shared_weights = [("language_model.embedding.word_embeddings.weight","lm_head_weight")]  # parameter sharing between stages
    model = Varuna( model, args.stage_to_rank_map, dry_run_input, global_batch_size, 
                        args.chunk_size, args.fp16, local_rank=args.local_rank, 
                        device=args.local_rank, shared_weights=shared_weights)

    # now model is a subset of the original model, moved to the GPU on each process

    optimizer = get_optimizer(model)
    model.set_optimizer(optimizer)

~~~~

### Training loop.

The Varuna training loop does not require a separate forward & backward step, the script may just call the `step` function. The input to this function should be of the per-process batch size (batch_size / data_parallel_workers), and should be a dictionary with arg names and values. The step function makes micro-batches out of this input batch, completes the fwd/bwd pipeline schedule and reduces the gradients/overflow over the whole job, returning the loss and overflow boolean. 

~~~~

inputs = dict({
    "input_ids": tokens,
    "position_ids": position_ids,
    "attention_mask": attention_mask,
    "loss_mask": loss_mask,
    "labels": labels
})

loss, overflow = model.step(inputs)
loss = torch.Tensor([loss])

if not overflow:
  optimizer.step()

~~~~

### Launcher and Arguments


To launch a distributed training process using Varuna, use the run_varuna.py script as follows:

python -m varuna.run_varuna --machine_list <file_with_ips> --gpus_per_node <num_gpus_per_node> --batch-size <total_effective_batch_size> --nstages <number_of_pipeline_stages> --chunk_size <micro_batch_size_for_pipeline> --code_dir <working_dir_for_training> user_training_script.py <...user args...>

See [Launching varuna](docs/html/launching.html).

This expects all machines in the machine_list to be set up with necessary code/libraries in code_dir and have gpus_per_node GPUs working. The job is launched with all workers running the user_training_script and args.

This launcher passes a few arguments to the user training script for Varuna. These should be passed during `Varuna` initialisation in the python script:
* rank: process rank in overall distributed job
* local_rank: process rank in the local node 
* stage_to_rank_map: varuna config info about stage placement
* chunk_size: micro batch size for Varuna pipeline
* batch-size: per process batch size

### Changing resources: job morphing

Varuna enables training on a changing set of nodes/gpus. This is through monitoring the machine_list text file of IPs with the set of available nodes at any time. 
Training jobs are launched from a long-living manager.
On detecting a change, Varuna checkpoints, stops and relaunches the job from the manager. To allow for this on-demand checkpoint/stop, varuna relies on user signals (SIGUSR1 in unix). The user therefore needs to add a simple handler for this signal to their training script.
See [Morphing](docs/html/morphing.html).

~~~~
if __name__ == "__main__":

    def handler(signum,_):
        save_checkpoint(iteration, model, optimizer, lr_scheduler)
        exit()

    signal.signal(signal.SIGUSR1, handler)
~~~~

<!-- 
- change polling, reduce freq, get_available
- manager ip argument
- checkpoint folder -->

### Profiling, config selection

Varuna supports auto-configuration of data-parallel and pipeline-parallel dimensions that saves the user from running and comparing different configs for better performance. To enable this, the user needs to run one-time profiling of the model and network conditions using the `Profiler` class in Varuna.
See [Varuna Profiling](docs/html/profiler.html).
This is instantiated similar to Varuna and runs as a distributed process:

~~~~
model = BuildModel()
profiler = Profiler(model, args.device, fp16=args.fp16)

def get_batch(size):
  # function to get sample batches of given size for profiling
  return batch

profiler.initialize(get_batch)
microbatches_to_profile = list(range(1,max_micro_BS))
profile = profiler.profile_all(get_batch, microbatches_to_profile, out_folder=args.save)

~~~~

Each process profiles the compute of different cutpoints and at the same time measures communication with other processes. This builds and saves a profile of the model in the specified location, from where it can be accessed by the `AutoConfig` class. `AutoConfig` calculates the different configs for a given number of GPUs and simulates them using information from the pre-built profile to compare and return the best performing setting in a few seconds. This calculation is triggered by run_varuna when no nstages and chunk_size arguments are given and a profile location is passed.
