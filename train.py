import argparse
from datasets import load_dataset
from arc.arc import ARCSolver

def main():
    parser = argparse.ArgumentParser(description='Train ARCSolver with ARC dataset')
    parser.add_argument('--token', type=str, default=None, help='HuggingFace token')
    parser.add_argument('--dataset', type=str, default='/home/student/workspace/dataset', help='Dataset name or path')
    args = parser.parse_args()
    
    print("Initializing model...")
    solver = ARCSolver(token=args.token)
    
    print("Starting training...")
    solver.train(train_dataset_path=args.dataset)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
