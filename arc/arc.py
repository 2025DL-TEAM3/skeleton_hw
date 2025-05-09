from transformers import GenerationConfig
import torch
from typing import List, Dict, Any
import numpy as np
import copy

from .utils import (
    system_prompt,
    user_message_template1,
    user_message_template2,
    user_message_template3,
)  # Ensure these are updated if prompt engineering changes
from .augmentations import apply_random_augmentation

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
import os
import json
from torch import nn
import torch.nn.functional as F
import glob
import random

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader


class ARCSolver:
    def __init__(
        self,
        token=None,
        model_id="Qwen/Qwen3-4B",
        artifacts_path="artifacts/checkpoint-final",
    ):
        self.model_id = model_id
        self.artifacts_path = artifacts_path
        self.token = token
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._load_model_and_tokenizer()

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]

        self.original_lora_weights = None

    def _load_model_and_tokenizer(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
            use_cache=False,
            device_map=self.device,
            token=self.token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}

        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
        """
        Format 2D grid into LLM input tokens

        Args:
            grid (List[List[int]]): 2D grid

        Returns:
            ids (List[int]): Token list for LLM
        """
        ids = []

        for row in grid:
            for col in row:
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def grid_to_str(self, grid: List[List[int]]) -> str:
        return "\n".join("".join(map(str, r)) for r in grid)

    def format_prompt(self, datapoint: Dict[str, Any], is_ttt_prompt: bool = False):
        """
        Args:
            datapoint (dict): contains training data, test input

        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """
        train_examples = datapoint["train"]

        if is_ttt_prompt:
            test_inp = datapoint["train"][-1]["input"]

        else:
            test_inp = datapoint["test"][0]["input"]

        current_system_prompt = "You are an expert ARC puzzle solver. Analyze the provided input-output examples to understand the underlying transformation rule. Then, apply this rule to the final input grid to generate the correct output grid. Represent grids as rows of numbers, with each row on a new line."

        messages = [
            {"role": "system", "content": current_system_prompt},
        ]

        current_user_message_template1 = (
            "Here are some examples of a transformation rule:"
        )
        messages.append({"role": "user", "content": current_user_message_template1})

        for ex in train_examples:
            msg_content = f"Example Input:\n{self.grid_to_str(ex['input'])}\nExample Output:\n{self.grid_to_str(ex['output'])}"
            messages.append({"role": "user", "content": msg_content})

        if not is_ttt_prompt:
            current_user_message_template2 = (
                "Now, based on these examples, transform the following input grid:"
            )
            current_user_message_template3 = "Provide only the output grid."
            test_msg_content = f"{current_user_message_template2}\nInput:\n{self.grid_to_str(test_inp)}\n{current_user_message_template3}"
            messages.append({"role": "user", "content": test_msg_content})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=not is_ttt_prompt,
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        return {
            "input_ids": input_ids[0],
            "input_grid_for_shape_ref": test_inp,
            "train_examples_for_shape_ref": train_examples,
        }

    def seq2seq_loss(self, prompt_ids, target_grid_ids):
        inp = torch.cat([prompt_ids.unsqueeze(0), target_grid_ids.unsqueeze(0)], dim=1)
        labels = inp.clone()
        labels[:, : prompt_ids.size(0)] = -100  # Mask prompt part
        outputs = self.model(input_ids=inp, labels=labels)
        return outputs.loss

    def _perform_ttt(
        self,
        ttt_examples: List[Dict[str, Any]],
        ttt_steps: int = 5,
        ttt_lr: float = 1e-4,
    ):
        if not ttt_examples:
            return

        self.model.train()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=ttt_lr
        )

        for step in range(ttt_steps):
            total_loss_this_step = 0
            num_samples_this_step = 0
            for example_pair in ttt_examples:
                ttt_datapoint = {
                    "train": [example_pair],
                    "test": [{"input": example_pair["input"]}],
                }

                prompt_messages_for_ttt = [
                    {
                        "role": "system",
                        "content": "You are learning to map this input to this output.",
                    },
                    {
                        "role": "user",
                        "content": f"Input:\n{self.grid_to_str(example_pair['input'])}\nOutput:",
                    },
                ]
                prompt_ids = self.tokenizer.apply_chat_template(
                    prompt_messages_for_ttt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                ).to(self.device)[0]

                target_grid_tokens = self.format_grid(example_pair["output"])
                target_ids = torch.tensor(target_grid_tokens, dtype=torch.long).to(
                    self.device
                )

                if target_ids.nelement() == 0:
                    continue

                loss = self.seq2seq_loss(prompt_ids, target_ids)
                if loss is not None:
                    loss.backward()
                    total_loss_this_step += loss.item()
                    num_samples_this_step += 1

            if num_samples_this_step > 0:
                optimizer.step()
                optimizer.zero_grad()

        self.model.eval()

    def train(
        self,
        train_dataset_path: str,
        epochs: int = 3,
        lr: float = 5e-5,
        batch_size: int = 1,
        grad_accum_steps: int = 4,
        max_steps: int = -1,
    ):
        self.model.train()

        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],  # Common for Llama/Qwen
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        dataset = ARCDataset(train_dataset_path, self.tokenizer, self, augment=False)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn
        )

        optimizer = AdamW(self.model.parameters(), lr=lr)

        global_step = 0
        for epoch in range(epochs):
            print(f"Starting Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(dataloader):
                if max_steps > 0 and global_step >= max_steps:
                    break

                input_ids_batch = batch["input_ids"].to(self.device)
                target_ids_batch = batch["target_ids"].to(self.device)
                attention_mask_batch = batch["attention_mask"].to(self.device)

                labels_batch = input_ids_batch.clone()

                combined_input_ids = []
                labels = []
                max_len = 0

                for i in range(input_ids_batch.size(0)):
                    prompt_len = (
                        (input_ids_batch[i] != self.tokenizer.pad_token_id).sum().item()
                    )
                    target_len = (
                        (target_ids_batch[i] != self.tokenizer.pad_token_id)
                        .sum()
                        .item()
                    )

                    current_combined = torch.cat(
                        [
                            input_ids_batch[i, :prompt_len],
                            target_ids_batch[i, :target_len],
                        ]
                    )
                    current_labels = torch.full_like(current_combined, -100)
                    current_labels[prompt_len:] = target_ids_batch[i, :target_len]

                    combined_input_ids.append(current_combined)
                    labels.append(current_labels)
                    if current_combined.size(0) > max_len:
                        max_len = current_combined.size(0)

                padded_combined_input_ids = torch.full(
                    (input_ids_batch.size(0), max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device,
                )
                padded_labels = torch.full(
                    (input_ids_batch.size(0), max_len),
                    -100,
                    dtype=torch.long,
                    device=self.device,
                )
                new_attention_mask = torch.zeros(
                    (input_ids_batch.size(0), max_len),
                    dtype=torch.long,
                    device=self.device,
                )

                for i in range(input_ids_batch.size(0)):
                    seq_len = combined_input_ids[i].size(0)
                    padded_combined_input_ids[i, :seq_len] = combined_input_ids[i]
                    padded_labels[i, :seq_len] = labels[i]
                    new_attention_mask[i, :seq_len] = 1

                outputs = self.model(
                    input_ids=padded_combined_input_ids,
                    labels=padded_labels,
                    attention_mask=new_attention_mask,
                )
                loss = outputs.loss

                if loss is not None:
                    loss = loss / grad_accum_steps
                    loss.backward()

                if (step + 1) % grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    print(
                        f"Epoch {epoch+1}, Step {global_step}, Loss: {loss.item() * grad_accum_steps:.4f}"
                    )

                global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                break

        os.makedirs(self.artifacts_path, exist_ok=True)
        self.model.save_pretrained(self.artifacts_path)
        self.tokenizer.save_pretrained(self.artifacts_path)
        print(f"Model and tokenizer saved to {self.artifacts_path}")
        self.model.eval()

    def predict(
        self,
        examples: List[Dict[str, Any]],
        question_input: List[List[int]],
        num_candidates: int = 3,
        ttt_steps: int = 0,
    ):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        self.model.eval()

        original_model_state_dict = None
        if ttt_steps > 0 and hasattr(
            self.model, "named_parameters"
        ):  # Check if model has parameters (e.g. not None)
            try:
                pass
            except Exception as e:
                print(f"Warning: Could not save original LoRA state for TTT: {e}")

        if ttt_steps > 0:
            print(f"Performing Test-Time Training for {ttt_steps} steps...")
            self._perform_ttt(examples, ttt_steps=ttt_steps)

        datapoint = {"train": examples, "test": [{"input": question_input}]}
        prompt_details = self.format_prompt(datapoint)
        input_ids = prompt_details["input_ids"].unsqueeze(0)

        generation_config = GenerationConfig(
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            num_return_sequences=num_candidates,
            early_stopping=True,
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
            )

        candidate_grids = []
        for i in range(num_candidates):
            output_tokens = outputs[i, input_ids.size(1) :].tolist()
            parsed_grid = self.parse_grid(output_tokens)
            candidate_grids.append(parsed_grid)

        final_grid = np.random.randint(0, 10, (3, 3))
        if not candidate_grids:
            return final_grid.tolist()

        ref_train_input = np.array(
            prompt_details["train_examples_for_shape_ref"][0]["input"]
        )
        ref_train_output = np.array(
            prompt_details["train_examples_for_shape_ref"][0]["output"]
        )
        ref_test_input = np.array(prompt_details["input_grid_for_shape_ref"])

        target_x, target_y = ref_test_input.shape
        if ref_train_input.shape == ref_train_output.shape:
            target_x, target_y = ref_test_input.shape
        elif ref_train_input.shape[0] != 0 and ref_train_input.shape[1] != 0:
            if (
                ref_train_output.shape[0] % ref_train_input.shape[0] == 0
                and ref_train_output.shape[1] % ref_train_input.shape[1] == 0
            ):
                scale_x = ref_train_output.shape[0] // ref_train_input.shape[0]
                scale_y = ref_train_output.shape[1] // ref_train_input.shape[1]
                target_x = ref_test_input.shape[0] * scale_x
                target_y = ref_test_input.shape[1] * scale_y
            else:
                target_x, target_y = (10, 10)
        else:
            target_x, target_y = (10, 10)

        for cand_grid_list in candidate_grids:
            try:
                if not cand_grid_list or not all(
                    isinstance(r, list) for r in cand_grid_list
                ):
                    continue
                if not cand_grid_list:
                    continue
                max_cols = 0
                if cand_grid_list and isinstance(cand_grid_list[0], list):
                    max_cols = (
                        max(len(r) for r in cand_grid_list if isinstance(r, list))
                        if cand_grid_list
                        else 0
                    )

                uniform_grid_list = []
                for r in cand_grid_list:
                    if isinstance(r, list):
                        uniform_grid_list.append(r + [0] * (max_cols - len(r)))

                if not uniform_grid_list:
                    continue

                grid_arr = np.array(uniform_grid_list)
                if grid_arr.size == 0:
                    continue

                if grid_arr.shape[0] >= target_x and grid_arr.shape[1] >= target_y:
                    final_grid = grid_arr[:target_x, :target_y]
                    break
                elif grid_arr.size >= target_x * target_y:
                    final_grid = np.resize(grid_arr, (target_x, target_y))
                    break
                else:
                    padded_arr = np.zeros((target_x, target_y), dtype=int)
                    min_rows = min(target_x, grid_arr.shape[0])
                    min_cols = min(target_y, grid_arr.shape[1])
                    if grid_arr.ndim == 2 and min_rows > 0 and min_cols > 0:
                        padded_arr[:min_rows, :min_cols] = grid_arr[
                            :min_rows, :min_cols
                        ]
                    final_grid = padded_arr
                    break
            except Exception as e:
                continue
        else:
            if candidate_grids and candidate_grids[0]:
                try:
                    first_cand_arr = np.array(candidate_grids[0])
                    if first_cand_arr.size > 0:
                        final_grid = np.resize(
                            first_cand_arr, (target_x, target_y if target_y > 0 else 1)
                        )
                    else:
                        final_grid = np.random.randint(
                            0, 10, (max(1, target_x), max(1, target_y))
                        )
                except:
                    final_grid = np.random.randint(
                        0, 10, (max(1, target_x), max(1, target_y))
                    )
            else:
                final_grid = np.random.randint(
                    0, 10, (max(1, target_x), max(1, target_y))
                )

        return final_grid.tolist()

    def prepare_evaluation(self):
        self._load_model_and_tokenizer()
        try:
            self.model = PeftModel.from_pretrained(
                self.model, self.artifacts_path, is_trainable=False
            )
            print(
                f"Successfully loaded LoRA adapter from {self.artifacts_path} for evaluation."
            )
        except Exception as e:
            print(
                f"No LoRA adapter found at {self.artifacts_path} or error loading: {e}. Using base model for evaluation."
            )
        self.model.eval()


class ARCDataset(Dataset):
    """
    Dataset class for ARC training examples.
    """

    def __init__(
        self,
        dataset_path_or_files: Any,
        tokenizer: AutoTokenizer,
        solver,
        augment: bool = False,
        max_examples_per_task: int = 100,
        steps_per_file: int = 50,
    ):
        self.tokenizer = tokenizer
        self.solver = solver
        self.augment = augment
        self.max_examples_per_task = max_examples_per_task
        self.steps_per_file = steps_per_file

        # Determine the data files
        if isinstance(dataset_path_or_files, str):
            if os.path.isdir(dataset_path_or_files):
                data_files = glob.glob(f"{dataset_path_or_files}/*.json")
            elif os.path.isfile(
                dataset_path_or_files
            ) and dataset_path_or_files.endswith(".json"):
                data_files = [dataset_path_or_files]
            else:
                raise ValueError(f"Invalid dataset path: {dataset_path_or_files}")
        elif isinstance(dataset_path_or_files, list):
            data_files = dataset_path_or_files
        else:
            print(
                "No valid dataset paths provided. Expecting dataset objects or paths."
            )
            self.all_examples = []
            self.total_steps = 0
            return

        if not data_files:
            raise ValueError("No dataset files found or provided.")

        print(f"Loading {len(data_files)} JSON task files...")

        # Load data from all files
        self.all_examples = []

        for file_path in data_files:
            try:
                with open(file_path, "r") as f:
                    file_examples = json.load(f)
                    if isinstance(file_examples, list):
                        self.all_examples.append(
                            {"file_path": file_path, "examples": file_examples}
                        )
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue

        if not self.all_examples:
            raise ValueError("No valid examples found in the dataset files.")

        self.total_steps = len(self.all_examples) * self.steps_per_file
        print(
            f"Loaded {len(self.all_examples)} files with {self.total_steps} total steps."
        )

    def __len__(self):
        return self.total_steps

    def _sample_examples(self, examples: List[Dict], num_samples: int) -> List[Dict]:
        """
        Randomly sample a specified number of examples.
        If examples are fewer than num_samples, sample with replacement.
        """
        if len(examples) < num_samples:
            return random.choices(examples, k=num_samples)
        return random.sample(examples, num_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Fetch a single training example.
        Each step randomly selects a JSON file and samples 4 examples:
        - 3 for training
        - 1 for testing
        """
        # Randomly select a file
        file_data = random.choice(self.all_examples)
        examples = file_data["examples"]

        # Ensure we have enough examples
        if len(examples) < 4:
            additional_file = random.choice(self.all_examples)
            examples += additional_file["examples"]

        # Randomly sample 4 examples
        selected_examples = self._sample_examples(examples, 4)

        # Assign 3 for training, 1 for testing
        train_examples = selected_examples[:3]
        test_example = selected_examples[3]

        # Construct the datapoint
        datapoint = {
            "train": train_examples,
            "test": [{"input": test_example["input"]}],
        }

        # Generate prompt and target tensors
        prompt = self.solver.format_prompt(datapoint)
        input_ids = prompt["input_ids"]
        target_output = test_example["output"]
        target_tokens = self.solver.format_grid(target_output)
        target_ids = torch.tensor(target_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def collate_fn(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function to handle padding and tensor creation.
        """
        input_ids_list = [item["input_ids"] for item in batch]
        target_ids_list = [item["target_ids"] for item in batch]

        # Pad sequences
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_target_ids = torch.nn.utils.rnn.pad_sequence(
            target_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Create attention masks
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": padded_input_ids,
            "target_ids": padded_target_ids,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    solver = ARCSolver()

    # Example usage
