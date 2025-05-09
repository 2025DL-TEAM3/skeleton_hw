import argparse
import os
import numpy as np
from arc import ARCSolver
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import glob
import json
import random

def grid_accuracy(pred_grid, true_grid):
    """
    그리드 정확도 계산 - 모든 셀이 정확히 일치해야 정확한 것으로 간주
    """
    pred_shape = pred_grid.shape
    true_shape = np.array(true_grid).shape
    
    # 크기가 다르면 0 반환
    if pred_shape != true_shape:
        return 0
    
    # 모든 셀이 일치하는지 확인
    return np.array_equal(pred_grid, true_grid)

def cell_accuracy(pred_grid, true_grid):
    """
    셀 단위 정확도 계산 - 각 셀의 정확도를 평균
    """
    pred_np = np.array(pred_grid).flatten()
    true_np = np.array(true_grid).flatten()
    
    # 크기가 다르면 더 작은 쪽에 맞춤
    min_len = min(len(pred_np), len(true_np))
    pred_np = pred_np[:min_len]
    true_np = true_np[:min_len]
    
    # 셀 단위 정확도 계산
    return accuracy_score(true_np, pred_np)

def visualize_example(example_id, input_grid, true_output, pred_output):
    """결과 시각화"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 입력 그리드
    axes[0].imshow(input_grid, cmap='viridis', vmin=0, vmax=9)
    axes[0].set_title('Input')
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            axes[0].text(j, i, str(input_grid[i][j]), 
                         ha='center', va='center', color='white')
    
    # 실제 출력
    true_array = np.array(true_output)
    axes[1].imshow(true_array, cmap='viridis', vmin=0, vmax=9)
    axes[1].set_title('True Output')
    for i in range(len(true_array)):
        for j in range(len(true_array[0])):
            axes[1].text(j, i, str(true_array[i][j]), 
                         ha='center', va='center', color='white')
    
    # 예측 출력
    axes[2].imshow(pred_output, cmap='viridis', vmin=0, vmax=9)
    axes[2].set_title('Predicted Output')
    for i in range(len(pred_output)):
        for j in range(len(pred_output[0])):
            axes[2].text(j, i, str(pred_output[i][j]), 
                         ha='center', va='center', color='white')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/example_{example_id}.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate ARCSolver on ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', 
                        help='Dataset path')
    parser.add_argument('--num_examples', type=int, default=50, 
                        help='Number of examples to evaluate')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize predictions')
    args = parser.parse_args()
    
    # 환경 변수 설정
    os.environ['DATASET_PATH'] = args.dataset
    
    # JSON 파일 목록 가져오기
    print("데이터셋 로드 중...")
    data_files = glob.glob(f"{args.dataset}/*.json")
    if not data_files:
        raise ValueError(f"No JSON files found in {args.dataset}")
    
    print(f"총 {len(data_files)}개의 JSON 파일을 찾았습니다.")
    
    # 모든 JSON 파일을 로드하여 메모리에 저장 (ARCDataset 클래스와 동일한 방식)
    all_examples = []
    for file_path in data_files:
        try:
            with open(file_path, 'r') as f:
                file_examples = json.load(f)
                if isinstance(file_examples, list) and len(file_examples) > 0:
                    all_examples.append({
                        "file_path": file_path,
                        "examples": file_examples
                    })
        except Exception as e:
            print(f"파일 로드 중 오류 발생: {file_path} - {e}")
    
    if not all_examples:
        raise ValueError("유효한 예제가 포함된 JSON 파일이 없습니다.")
    
    print(f"총 {len(all_examples)}개의 JSON 파일이 성공적으로 로드되었습니다.")
    
    # ARCSolver 인스턴스 생성 및 학습된 모델 로드
    print("모델 초기화 중...")
    solver = ARCSolver(token=args.token)
    solver.prepare_evaluation()
    
    # 결과 저장용 변수
    results = []
    grid_accuracies = []
    cell_accuracies = []
    
    # 재현성을 위한 시드 설정
    random.seed(42)
    
    # 평가할 파일 수 설정
    num_files = min(args.num_examples, len(all_examples))
    print(f"총 {num_files}개의 파일에서 예제를 평가합니다.")
    
    # 랜덤하게 파일 선택
    selected_files = random.sample(all_examples, num_files)
    
    for i, file_data in enumerate(selected_files):
        file_path = file_data["file_path"]
        file_examples = file_data["examples"]
        file_name = os.path.basename(file_path)
        
        print(f"\n파일 평가 중: {file_name}")
        
        # 최소 4개의 예제가 있어야 함
        if len(file_examples) < 4:
            print(f"예제가 부족합니다 ({len(file_examples)}개): {file_path}")
            continue
        
        # 4개의 예제 랜덤 선택
        selected_examples = random.sample(file_examples, 4)
        
        # 3개는 학습용, 1개는 테스트용으로 분리
        train_examples = selected_examples[:3]
        test_example = selected_examples[3]
        
        input_grid = test_example["input"]
        true_output = test_example["output"]
        
        try:
            # 예측 실행
            pred_output = solver.predict(train_examples, input_grid)
            
            # 정확도 계산
            g_acc = grid_accuracy(pred_output, true_output)
            c_acc = cell_accuracy(pred_output, true_output)
            
            grid_accuracies.append(g_acc)
            cell_accuracies.append(c_acc)
            
            result = {
                "file": file_name,
                "grid_accuracy": g_acc,
                "cell_accuracy": c_acc,
                "input_shape": np.array(input_grid).shape,
                "true_output_shape": np.array(true_output).shape,
                "pred_output_shape": pred_output.shape
            }
            
            results.append(result)
            
            print(f"그리드 정확도: {g_acc}, 셀 정확도: {c_acc:.4f}")
            print(f"입력 크기: {np.array(input_grid).shape}, "
                  f"실제 출력 크기: {np.array(true_output).shape}, "
                  f"예측 출력 크기: {pred_output.shape}")
            
            # 결과 시각화
            if args.visualize:
                visualize_example(file_name, input_grid, true_output, pred_output)
            
        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            continue
    
    # 전체 정확도 계산 및 출력
    if results:
        avg_grid_acc = np.mean(grid_accuracies)
        avg_cell_acc = np.mean(cell_accuracies)
        
        print("\n===== 평가 결과 =====")
        print(f"평가한 예제 수: {len(results)}")
        print(f"평균 그리드 정확도: {avg_grid_acc:.4f}")
        print(f"평균 셀 정확도: {avg_cell_acc:.4f}")
        
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        with open('results/evaluation_results.json', 'w') as f:
            json.dump({
                "num_examples": len(results),
                "avg_grid_accuracy": float(avg_grid_acc),
                "avg_cell_accuracy": float(avg_cell_acc),
                "detailed_results": results
            }, f, indent=2)
        
        print("결과가 results/evaluation_results.json에 저장되었습니다.")
    else:
        print("평가할 유효한 예제가 없습니다.")

if __name__ == "__main__":
    main()
