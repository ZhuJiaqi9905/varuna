
from megatron.arguments import parse_args
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron import get_tokenizer
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.model import GPT2Model
from megatron.model import get_params_for_weight_decay_optimization
from megatron.global_vars import set_global_variables
from megatron import get_args
from varuna import Profiler
from apex.optimizers import FusedLAMB as LAMB

import torch

from apex import amp
from apex.amp import _amp_state

set_global_variables(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
args = get_args()

max_micro_BS = 50

train_val_test_num_samples = [ 5 * max_micro_BS, 0, 0 ]

model = GPT2Model(num_tokentypes=0, parallel_output=True)
profiler = Profiler(model, args.fp16)

# Build the datasets.
train_ds, _test , _valid = build_train_valid_test_datasets(
                            data_prefix=args.data_path,
                            data_impl=args.data_impl,
                            splits_string=args.split,
                            train_valid_test_num_samples=train_val_test_num_samples,
                            seq_length=args.seq_length,
                            seed=args.seed,
                            skip_warmup=(not args.mmap_warmup))


def get_batch(size, device=None):
    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=size)
    dry_run_input = next(iter(dataloader))

    # Unpack.
    tokens_ = dry_run_input['text'].long()
    labels = tokens_[:,1:].contiguous()
    tokens = tokens_[:,:-1].contiguous()

    # Get the masks and postition ids.
    tokenizer = get_tokenizer()
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    dry_run_input = dict({
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "labels": labels
    })

    if device is not None:
        for k in dry_run_input:
            dry_run_input[k] = dry_run_input[k].to(device)
    
    return dry_run_input

profiler.initialize(get_batch(1) , from_cache=False)

# with open(profile_name,"w") as f:
#     f.write(("Model memory: " + str(model_mem) + "\n"))

param_groups = get_params_for_weight_decay_optimization(model)
optimizer = LAMB(param_groups, lr=args.lr, weight_decay=args.weight_decay)
if args.fp16:
    if args.dynamic_loss_scale:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale="dynamic",min_loss_scale=args.min_scale)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**15
    else:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2", loss_scale=args.loss_scale, min_loss_scale=args.min_scale)

profile = profiler.profile(get_batch,[1]+ list(range(1,max_micro_BS)), optimizer)
 