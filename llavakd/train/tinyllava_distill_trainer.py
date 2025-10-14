import os
import torch
from torch import nn
import deepspeed
import torch.nn.functional as F
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from typing import List, Optional

from ..utils.train_utils import *
from ..utils.distill_loss_utils import skewed_forward_kl, skewed_reverse_kl, skewed_reverse_kl_womask
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)
import wandb

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class DistillLLaVATrainer(Trainer):

    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.loss_fct =  nn.KLDivLoss(reduction="batchmean")
        self.niter = 0
        self.mismatch_count = 0

        self.teacher_model = self.teacher_model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)

        self.model = self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)


    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },

                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    }
                ]

            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer


    def compute_loss(self, model, inputs, return_outputs=False):

        self.niter += 1
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        with torch.no_grad():
            teacher_outputs, multimodal_teacher_labels, teacher_image_rela, _, teacher_rela, _, _ = self.teacher_model(**inputs)
        # soft_label = inputs.pop('soft_label')
        outputs, multimodal_labels, student_image_rela, _, student_rela,  num_images, spilt_sizes  = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)


        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

        loss = outputs['loss']

        _, _, voc_size = outputs['logits'].shape
        shift_student_logits = outputs['logits'][..., :-1, :].contiguous()
        shift_teacher_logits = teacher_outputs['logits'][..., :-1, :].contiguous()
        
        shift_labels = multimodal_labels[..., 1:].contiguous()
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(outputs['logits'].device)
        
        shift_teacher_labels = multimodal_teacher_labels[..., 1:].contiguous()
        shift_teacher_labels = shift_teacher_labels.view(-1)
        shift_teacher_labels = shift_teacher_labels.to(outputs['logits'].device)


        # # only Answers KL distill loss 
        mask = torch.ne(shift_labels, -100).int()
        teacher_mask = torch.ne(shift_teacher_labels, -100).int()

        masked_student_logits = shift_student_logits.view(-1, voc_size)[mask.bool()]       
        masked_teacher_logits = shift_teacher_logits.view(-1, voc_size)[teacher_mask.bool()]   

        if masked_teacher_logits.shape != masked_student_logits.shape:
            print(masked_teacher_logits.shape, masked_student_logits.shape)
            print(masked_teacher_logits.device, masked_student_logits.device)
            loss = outputs['loss']
        else:
            forward_distillation_loss = self.loss_fct(
                    nn.functional.log_softmax(masked_student_logits, dim=-1),
                    nn.functional.softmax(masked_teacher_logits, dim=-1)
                )

            loss = forward_distillation_loss + loss
            

        if num_images == 1:
            shift_STU_image_logits =  outputs['logits'][:, spilt_sizes[0][0] :spilt_sizes[0][0]+728]
            shift_Tea_image_logits =  teacher_outputs['logits'][:, spilt_sizes[0][0] :spilt_sizes[0][0]+728, :]
            forward_visual_distillation = self.loss_fct(
                    nn.functional.log_softmax(shift_STU_image_logits.view(-1, voc_size), dim=-1),
                    nn.functional.softmax(shift_Tea_image_logits.view(-1, voc_size), dim=-1)
                )

            student_img_rela = torch.matmul(shift_STU_image_logits, shift_STU_image_logits.permute(0,2,1))
            teacher_img_rela = torch.matmul(shift_Tea_image_logits, shift_Tea_image_logits.permute(0,2,1))
            
            llm_visual_rela_distill_loss = F.cosine_similarity(teacher_img_rela.view(-1).unsqueeze(0), student_img_rela.view(-1).unsqueeze(0))
            llm_visual_rela_distill_loss = 1 - llm_visual_rela_distill_loss.mean()

            loss = loss + forward_visual_distillation + llm_visual_rela_distill_loss

        elif num_images == 0:
            loss = loss
        else:
            raise NotImplementedError

            
        if self.niter % 100 == 0 and self.args.local_rank==0:
            wandb.log({
                "train/llm_loss": outputs['loss'],
            }, step=self.niter)

        return (loss, outputs) if return_outputs else loss
    

