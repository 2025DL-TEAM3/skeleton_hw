from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import (
    system_prompt,
    user_message_template1,
    user_message_template2,
    user_message_template3,
)
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
import os
import json
from torch import nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

import random

# --- Augmentation Helpers ---
def rotate_grid_augmentation(grid_list_list: List[List[int]], k: int) -> List[List[int]]:
    """Rotate grid by k * 90 degrees clockwise."""
    if not grid_list_list or not grid_list_list[0]:
        return grid_list_list
    return np.rot90(np.array(grid_list_list, dtype=int), k=-k).tolist()

def flip_grid_augmentation(grid_list_list: List[List[int]], axis: int) -> List[List[int]]:
    """Flip grid. axis 0 for UD, 1 for LR."""
    if not grid_list_list or not grid_list_list[0]:
        return grid_list_list
    return np.flip(np.array(grid_list_list, dtype=int), axis=axis).tolist()

def permute_colors_augmentation(grid_list_list: List[List[int]], color_map: dict[int, int]) -> List[List[int]]:
    """Permute colors in the grid using a color_map, keeping background (0) if not in map."""
    if not grid_list_list or not grid_list_list[0]:
        return grid_list_list
    return [[color_map.get(c, c) for c in row] for row in grid_list_list]

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "Qwen/Qwen3-4B"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,  # Allow the model to use custom code from the repository
            quantization_config=bnb_config,  # Apply the 4-bit quantization configuration
            attn_implementation="sdpa",  # Use scaled-dot product attention for better performance
            torch_dtype=torch.float16,  # Set the data type for the model
            use_cache=False,  # Disable caching to save memory
            device_map="cuda:0",  # Automatically map the model to available devices (e.g., GPUs)
            token=token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0]
            for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # 줄마다 012… 형식, 줄 끝에 \n
        return "\n".join("".join(str(c) for c in row) for row in grid)

    def format_prompt(self, datapoint):
        train_examples = datapoint["train"]
        test_inp = datapoint["test"][0]["input"]

        # ChatML 메시지 배열
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.append(f"{user_message_template1}\n")
        # 학습 예제 3개
        for ex in train_examples:
            msg = (
                f"input:\n{self.grid_to_str(ex['input'])}\n"
                f"output:\n{self.grid_to_str(ex['output'])}"
            )
            messages.append({"role": "user", "content": msg})

        test_msg = (
            f"{user_message_template2}\n"
            f"input:\n{self.grid_to_str(test_inp)}\n"
            f"{user_message_template3}"
        )
        messages.append({"role": "user", "content": test_msg})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # assistant 턴 여는 토큰 추가
            enable_thinking=False,  # THINKING MODE ON
            tokenize=True,
            return_tensors="pt",
        ).to(self.device)

        return {
            "input_ids": input_ids[0],  # [seq] → [L]
            "input": test_inp,
            "train": train_examples,
        }

    def _get_optimized_ttt_samples(self, original_examples: List[dict]) -> List[dict]:
        """
        Generate a minimal set of TTT samples for time-constrained inference.
        Optimized to create only 3-4 samples total, focusing on the most effective augmentations.
        """
        augmented_samples = [] # Each dict: {"context": [...], "ttt_input": grid, "ttt_output": grid}
        
        if not original_examples or len(original_examples) < 2:
            return []  # Need at least 2 examples for meaningful TTT
            
        # Select only the most effective transformations
        # Identity is always included, plus one rotation and one flip at most
        geometric_transforms = {
            "identity": lambda g: g,
            "rot90": lambda g: rotate_grid_augmentation(g, 1),
            "flip_lr": lambda g: flip_grid_augmentation(g, 1),
        }
        
        # Use only 1-2 examples for leave-one-out to minimize sample count
        num_examples_to_use = min(2, len(original_examples))
        selected_example_indices = random.sample(range(len(original_examples)), num_examples_to_use)
        
        for i in selected_example_indices:
            # Leave-One-Out: current example is the target for TTT
            ttt_target_example = original_examples[i]
            ttt_context_examples_orig = [ex for k, ex in enumerate(original_examples) if k != i]
            
            # Apply only identity transform to save time
            # This gives us the basic leave-one-out sample
            current_transformed_context = ttt_context_examples_orig.copy()
            augmented_samples.append({
                "context": current_transformed_context,
                "ttt_input": ttt_target_example["input"],
                "ttt_output": ttt_target_example["output"]
            })
            
            # Add just one geometric transform sample for diversity
            if len(augmented_samples) < 2:  # Limit to 2 samples max at this point
                transform_name = random.choice(["rot90", "flip_lr"])
                transform_func = geometric_transforms[transform_name]
                
                try:
                    # Transform context
                    geom_transformed_context = []
                    for ctx_ex in ttt_context_examples_orig:
                        geom_transformed_context.append({
                            "input": transform_func(ctx_ex["input"]),
                            "output": transform_func(ctx_ex["output"])
                        })
                    # Transform target
                    transformed_ttt_input = transform_func(ttt_target_example["input"])
                    transformed_ttt_output = transform_func(ttt_target_example["output"])
                    
                    augmented_samples.append({
                        "context": geom_transformed_context,
                        "ttt_input": transformed_ttt_input,
                        "ttt_output": transformed_ttt_output
                    })
                except Exception:
                    # Skip if transformation fails
                    pass
        
        # Add one color permutation sample if we have room (max 3 samples total)
        if len(augmented_samples) < 3 and len(original_examples) > 0:
            try:
                # Use the first example as target for simplicity
                color_target = original_examples[0]
                color_context = [ex for k, ex in enumerate(original_examples) if k != 0]
                if not color_context:
                    color_context = [color_target]  # Use target as context if it's the only example
                
                # Create a simple color permutation
                colors_to_permute = list(range(1, 10))  # Exclude background 0
                if len(colors_to_permute) > 1:
                    permuted_colors = random.sample(colors_to_permute, len(colors_to_permute))
                    color_map = {orig: perm for orig, perm in zip(colors_to_permute, permuted_colors)}
                    
                    color_perm_context = []
                    for ctx_ex in color_context:
                        color_perm_context.append({
                            "input": permute_colors_augmentation(ctx_ex["input"], color_map),
                            "output": permute_colors_augmentation(ctx_ex["output"], color_map)
                        })
                    color_perm_ttt_input = permute_colors_augmentation(color_target["input"], color_map)
                    color_perm_ttt_output = permute_colors_augmentation(color_target["output"], color_map)
                    
                    augmented_samples.append({
                        "context": color_perm_context,
                        "ttt_input": color_perm_ttt_input,
                        "ttt_output": color_perm_ttt_output
                    })
            except Exception:
                # Skip if color permutation fails
                pass
                
        # Ensure we don't have too many samples (max 4)
        if len(augmented_samples) > 4:
            augmented_samples = random.sample(augmented_samples, 4)
            
        return augmented_samples

    def stepwise_loss(self, prompt_ids, target_ids, use_cache=False):
        """
        prompt_ids : [1, L]  (프롬프트)
        target_ids : [1, T]  (정답 토큰 시퀀스)
        returns CE loss 합계 (scalar)
        """
        device = prompt_ids.device
        loss_total = 0.0
        past_kv = None  # KV-cache (use_cache=True 일 때만)

        for i in range(target_ids.size(1)):
            outputs = self.model(
                input_ids=prompt_ids if past_kv is None else target_ids[:, i - 1 : i],
                past_key_values=past_kv,
                use_cache=use_cache,
            )
            logits = outputs.logits[:, -1, :]  # 마지막 토큰 로짓
            loss_i = F.cross_entropy(logits, target_ids[:, i])
            loss_total += loss_i

            past_kv = outputs.past_key_values if use_cache else None
            # teacher token → 다음 입력
            prompt_ids = torch.cat([prompt_ids, target_ids[:, i : i + 1]], dim=1)

        return loss_total

    def seq2seq_loss(self, prompt_ids, target_ids):
        """
        prompt_ids  : [B, L]  ← 문제 설명(프롬프트)
        target_ids  : [B, T]  ← 정답 토큰 시퀀스
        ------------------------------------------
        inp   = [B, L+T]      ←  [프롬프트][정답] 한줄로 연결
        labels= same shape     (프롬프트 부분은 -100으로 마스킹)
        ------------------------------------------
        model(input_ids=inp, labels=labels)  →  .loss
        """
        eos = self.tokenizer.eos_token_id
        if target_ids[0, -1] != eos:
            eos_tensor = torch.tensor(
                [[eos]], dtype=target_ids.dtype, device=target_ids.device
            )
            target_ids = torch.cat([target_ids, eos_tensor], dim=1)

        inp = torch.cat([prompt_ids, target_ids], dim=1)
        labels = inp.clone()
        labels[:, : prompt_ids.size(1)] = -100

        outputs = self.model(input_ids=inp, labels=labels)
        return outputs.loss

    def train(self, train_dataset):
        """
        Train a model with train_dataset.
        """
        # Set the model to training mode
        self.model.train()

        # LoRA 설정 - Attention 모듈에 적용
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,  # LoRA 행렬의 랭크
            lora_alpha=32,  # LoRA 스케일링 파라미터
            lora_dropout=0.1,
            # Qwen2 모델의 트랜스포머 주요 가중치 타겟팅
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        # 모델에 LoRA 적용
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()  # 학습 가능한 파라미터 비율 출력

        dataset = ARCDataset(train_dataset, self.tokenizer, self)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # 메모리 효율성을 위한 그래디언트 누적 설정
        gradient_accumulation_steps = 4

        # Training loop
        epochs = 5
        global_step = 0
        for epoch in range(epochs):
            running = 0.0
            optimizer.zero_grad()  # 에포크 시작 시 그래디언트 초기화

            for step, batch in enumerate(dataloader):
                global_step += 1
                # 배치 데이터를 디바이스로 이동
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)

                # loss = self.stepwise_loss(input_ids, target_ids)
                loss = self.seq2seq_loss(input_ids, target_ids)

                # 역전파
                loss.backward()

                # 그래디언트 누적 후 최적화
                if global_step % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                # 손실 기록
                running += loss.item() * gradient_accumulation_steps

                # 로그 출력
                if step % 10 == 0:
                    print(
                        f"[E{epoch+1}] step {step} loss {loss.item()*gradient_accumulation_steps:.4f}"
                    )
                if global_step % 100 == 0:
                    print("Saving model...")
                    self.save_model()

            # 에포크 종료 시 평균 손실 출력
            print(f"Epoch {epoch+1} avg-loss {(running/len(dataloader)):.4f}")

        self.model.eval()  # Set model back to evaluation mode

    def save_model(self, data_path=None):
        if data_path is None:
            data_path = "artifacts/checkpoint-final_tmp"
        os.makedirs(data_path, exist_ok=True)
        self.model.save_pretrained(data_path)
        model_info = {
            "base_model": {
                "name": self.model.config._name_or_path,
                "type": self.model.config.model_type,
                "hidden_size": int(self.model.config.hidden_size),
                "vocab_size": int(self.tokenizer.vocab_size),
            }
        }
        with open(os.path.join(data_path, "model_config.json"), "w") as f:
            json.dump(model_info, f, indent=2)

    def predict(self, examples, questions_input):
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
        import time
        
        start_time = time.time()
        # --- BEGIN TTT (Test-Time Training) with Leave-One-Out and consistent formatting ---
        self.model.train()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        time_before_ttt = time.time()
        time_remaining = 50 - (time_before_ttt - start_time) 

        if time_remaining < 40:
            self.model.eval()  
            print(f"Warning: Skipping TTT due to time constraints. Only {time_remaining:.1f}s remaining.")
        else:
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            
            if trainable_params:
                optimizer = AdamW(trainable_params, lr=5e-7)
                
                ttt_max_steps = 8 
                
                ttt_samples = self._get_optimized_ttt_samples(examples)
                
                if ttt_samples:
                    for step_num in range(ttt_max_steps):
                        current_time = time.time()
                        time_used_so_far = current_time - start_time
                        
                        if time_used_so_far > 40:
                            print(f"TTT stopped early at step {step_num}/{ttt_max_steps} due to time constraints")
                            break
                            
                        ttt_sample = random.choice(ttt_samples)
                        
                        ttt_datapoint = {
                            "train": ttt_sample["context"],
                            "test": [{"input": ttt_sample["ttt_input"]}]
                        }
                        prompt_data = self.format_prompt(ttt_datapoint)
                        input_ids = prompt_data["input_ids"].unsqueeze(0).to(self.device)
                        
                        target_grid_tokens = self.format_grid(ttt_sample["ttt_output"])
                        if self.tokenizer.eos_token_id is not None:
                            target_grid_tokens.append(self.tokenizer.eos_token_id)
                        target_ids = torch.tensor([target_grid_tokens], device=self.device)
                        
                        optimizer.zero_grad()
                        loss = self.seq2seq_loss(input_ids, target_ids)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
        self.model.eval()
        # --- END TTT ---
        datapoint = {"train": examples, "test": [{"input": questions_input}]}

        prompt = self.format_prompt(datapoint)
        # input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)
        input_ids = prompt["input_ids"].unsqueeze(0)

        attn_mask = torch.ones_like(input_ids)

        # Qwen3 모델은 더 많은 토큰을 생성할 수 있도록 설정
        config = GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=20,  # 권장 값
            bos_token_id=151643,  # Qwen3 모델의 내부 기본값 명시적 사용
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            max_new_tokens=150,
            do_sample=True,
        )

        output = (
            self.model.generate(
                input_ids=input_ids,
                generation_config=config,
                attention_mask=attn_mask,
            )
            .squeeze()
            .cpu()
        )
        N_prompt = input_ids.size(1)

        output = output[N_prompt:].tolist()
        train_input = np.array(prompt["train"][0]["input"])
        train_output = np.array(prompt["train"][0]["output"])
        test_input = np.array(prompt["input"])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = train_output.shape[0] * test_input.shape[0] // train_input.shape[0]
            y = train_output.shape[1] * test_input.shape[1] // train_input.shape[1]

        try:
            grid = np.array(self.parse_grid(output))
            # grid = grid[:x, :y]

        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        # LoRA 어댑터 로드
        try:
            peft_config = PeftConfig.from_pretrained("artifacts/checkpoint-final")
            self.model = PeftModel.from_pretrained(
                self.model, "artifacts/checkpoint-final", is_trainable=True
            )
            print("Loaded LoRA adapter")
        except Exception as e:
            print(f"No LoRA adapter found or incompatible: {e}")

        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()


# ARCDataset 클래스 정의
class ARCDataset(Dataset):
    """Dataset for ARC training examples"""

    def __init__(self, examples, tokenizer, solver):
        """
        Initialize the ARC dataset

        Args:
            examples: Dataset object or path to dataset files
            tokenizer: Tokenizer for encoding inputs
            solver: ARCSolver instance
        """
        import glob
        import os
        import json
        import random

        self.tokenizer = tokenizer
        self.solver = solver

        # 데이터셋 파일 경로 가져오기
        if hasattr(examples, "data_files") and examples.data_files:
            data_files = examples.data_files
        else:
            # 기본 데이터셋 경로 사용
            dataset_path = os.environ.get(
                "DATASET_PATH", "/home/student/workspace/dataset"
            )
            data_files = glob.glob(f"{dataset_path}/*.json")

        if not data_files:
            raise ValueError("데이터셋 파일이 없습니다. 경로를 확인해주세요.")

        print(f"총 {len(data_files)}개의 JSON 파일을 로드합니다.")

        # 모든 JSON 파일을 로드하여 메모리에 저장
        self.all_examples = []
        for file_path in data_files:
            try:
                with open(file_path, "r") as f:
                    file_examples = json.load(f)
                    if isinstance(file_examples, list) and len(file_examples) > 0:
                        self.all_examples.append(
                            {"file_path": file_path, "examples": file_examples}
                        )
            except Exception as e:
                print(f"파일 로드 중 오류 발생: {file_path} - {e}")

        if not self.all_examples:
            raise ValueError("유효한 예제가 포함된 JSON 파일이 없습니다.")

        print(f"총 {len(self.all_examples)}개의 JSON 파일이 성공적으로 로드되었습니다.")

        # 총 학습 스텝 수 계산
        self.steps_per_file = 50  # 각 파일당 샘플링할 스텝 수
        self.total_steps = len(self.all_examples) * self.steps_per_file
        print(f"총 학습 스텝 수: {self.total_steps}")

    def __len__(self):
        """총 학습 스텝 수 반환"""
        return self.total_steps

    def __getitem__(self, idx):
        """
        각 학습 스텝에 대한 데이터 샘플 반환

        각 스텝에서는:
        1. 랜덤하게 JSON 파일 선택
        2. 해당 파일에서 4개의 예제 선택
        3. 3개는 학습 예제로, 1개는 테스트 예제로 사용
        """

        # 1. 랜덤하게 JSON 파일 하나 선택
        file_data = random.choice(self.all_examples)
        examples = file_data["examples"]

        if len(examples) < 4:
            # 예제가 4개 미만인 경우, 다른 파일에서 추가 예제 가져오기
            if len(self.all_examples) > 1:
                other_files = [f for f in self.all_examples if f != file_data]
                additional_file = random.choice(other_files)
                examples = examples + additional_file["examples"]

            # 그래도 4개가 안되면 중복 사용
            while len(examples) < 4:
                examples = examples + examples

        # 2. 4개의 예제 랜덤 선택
        selected_examples = random.sample(examples, 4)

        # 3. 3개는 학습용, 1개는 테스트용으로 분리
        train_examples = selected_examples[:3]
        test_example = selected_examples[3]

        # 4. 데이터포인트 구성
        datapoint = {
            "train": train_examples,  # 3개의 입력/출력 예제
            "test": [{"input": test_example["input"]}],  # 테스트할 입력
        }

        # 5. 포맷 및 텐서 변환
        prompt = self.solver.format_prompt(datapoint)
        # input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long)
        if isinstance(prompt["input_ids"], torch.Tensor):
            input_ids = prompt["input_ids"].clone().detach()
        else:
            input_ids = torch.tensor(prompt["input_ids"], dtype=torch.long)

        # 실제 정답 (타겟)
        target_output = test_example["output"]
        target_tokens = self.solver.format_grid(target_output)
        target_ids = torch.tensor(target_tokens, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }
