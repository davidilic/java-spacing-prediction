
import argparse
from pathlib import Path
from dataset.processing import DatasetCreator, DatasetManager
from dataset.visualization import DatasetVisualizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset for space/newline prediction.')
    parser.add_argument('repo_path', help='Path to repository containing .java files')
    parser.add_argument('train_path', help='Path to training data output')
    parser.add_argument('test_path', help='Path to test data output')
    parser.add_argument('--max_examples', type=int, default=50000, help='Maximum examples to collect')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of examples to use for testing')
    args = parser.parse_args()

    creator = DatasetCreator()
    manager = DatasetManager(creator, args.max_examples, args.test_ratio)
    visualizer = DatasetVisualizer()

    manager.process_repository(Path(args.repo_path))
    manager.save_datasets(Path(args.train_path), Path(args.test_path))

    visualizer.plot_distributions(Path(args.train_path))