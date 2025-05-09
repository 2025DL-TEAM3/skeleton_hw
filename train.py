import argparse
from datasets import load_dataset
from arc.arc import ARCSolver

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', help='Dataset name or path')
    args = parser.parse_args()
    
    # 데이터셋 로드
    print("Loading dataset...")
    dataset = load_dataset('json', data_files=f"{args.dataset}/*.json", split='train')
    
    # ARCSolver 인스턴스 생성
    print("Initializing model...")
    solver = ARCSolver(token=args.token)
    
    # 모델 학습
    print("Starting training...")
    solver.train(dataset)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
