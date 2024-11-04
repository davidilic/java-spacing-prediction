
import jsonlines
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path

class DatasetVisualizer:
    def plot_distributions(self, data_path: Path, save_path: Optional[Path] = None) -> None:
        """Plot distributions of spaces and newlines in the dataset."""
        try:
            distributions = self._collect_distributions(data_path)
            self._create_plots(distributions, save_path)
        except Exception as e:
            print(f"Error creating plots: {e}")
            
    def _collect_distributions(self, data_path: Path) -> Dict[str, List[int]]:
        """Collect space and newline distributions from data."""
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        distributions = {
            'spaces': [],
            'newlines': [],
        }
        
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                token_count = len(obj['tokens'])
                distributions['spaces'].extend(obj['spaces'])
                distributions['newlines'].extend([obj['newlines']] * token_count)
                
        return distributions
    
    def _create_plots(self, distributions: Dict[str, List[int]], save_path: Optional[Path] = None) -> None:
        """Create plots for space and newline distributions."""
        
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        axs[0].hist(distributions['spaces'], bins=50, color='skyblue', edgecolor='black')
        axs[0].set_title('Spaces Distribution')
        axs[0].set_xlabel('Number of spaces')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(distributions['newlines'], bins=50, color='salmon', edgecolor='black')
        axs[1].set_title('Newlines Distribution')
        axs[1].set_xlabel('Number of newlines')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            
        plt.show()
