import random
import re
import jsonlines
import hashlib
from dataclasses import dataclass, field
from typing import List, Set, Dict, Tuple, Optional
from pathlib import Path
from dataset.types import CodeInstance, TokenType

class JavaTokenizer:

    KEYWORDS = {
        'package', 'import', 'class', 'public', 'private', 'protected',
        'static', 'final', 'abstract', 'synchronized', 'return', 'void',
        'try', 'catch', 'throws', 'throw', 'if', 'else', 'while', 'for',
        'new', 'switch', 'case', 'default', 'break', 'continue',
        'implements', 'extends', 'interface', 'super', 'this'
    }
    
    TOKEN_PATTERN = re.compile(
        r'("[^"]*")|([a-zA-Z_]\w*)|([.,(){}[\];])|(\s+)|(\+|\-|\*|\/|\=|\>|\<|\!|\&|\|)'
    )

    @classmethod
    def get_token_type(cls, token: str) -> TokenType:
        if token in cls.KEYWORDS:
            return TokenType.KEYWORD
        elif token.startswith('"') or token.startswith("'"):
            return TokenType.STRING
        elif token in '.,(){}[];':
            return TokenType.DELIMITER
        elif token in '+-*/=><!&|':
            return TokenType.OPERATOR
        elif token.isspace():
            return TokenType.WHITESPACE
        return TokenType.IDENTIFIER

    @classmethod
    def tokenize(cls, code: str) -> List[Tuple[str, TokenType]]:
        code = cls.remove_comments(code)
        tokens = [t for t in cls.TOKEN_PATTERN.findall(code) if any(g for g in t)]
        return [(group, cls.get_token_type(group)) for t in tokens for group in t if group]

    @staticmethod
    def remove_comments(code: str) -> str:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

@dataclass
class LineState:

    """Holds the current state of line processing"""
    current_line: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    types: List[TokenType] = field(default_factory=list)
    spaces: List[int] = field(default_factory=list)
    depth: int = 0
    next_depth: int = 0
    space_count: int = 0
    newline_count: int = 0

    def reset(self, space_count: int = 0) -> None:
        """Reset line state while preserving depth"""
        self.current_line.clear()
        self.tokens.clear()
        self.types.clear()
        self.spaces.clear()
        self.space_count = space_count
        self.newline_count = 0
        self.depth = self.next_depth

    def update_whitespace(self, token: str) -> None:
        """Update whitespace counts from token"""
        self.space_count += token.count(' ') + token.count('\t')
        self.newline_count += token.count('\n')

    def add_token(self, token: str, token_type: TokenType) -> None:
        """Add a new token to the line state"""
        self.current_line.append(token)
        self.tokens.append(token)
        self.types.append(token_type)
        self.spaces.append(self.space_count)

    def update_depth(self, token: str) -> None:
        """Update scope depth based on braces"""
        if token == '{':
            self.next_depth += 1
        elif token == '}':
            self.next_depth = max(0, self.next_depth - 1)

class DatasetCreator:
    """Creates code instances from Java files."""
    def __init__(self, max_spaces: int = 48, max_newlines: int = 5):
        self.max_spaces = max_spaces
        self.max_newlines = max_newlines
        self.line_hashes: Set[int] = set()
        
    def process_file(self, filepath: Path, max_examples: int, current_count: int) -> Tuple[List[CodeInstance], bool]:
        """Process a single file and create code instances."""
        code = filepath.read_text(encoding='utf-8')
        file_id = self._generate_file_id(code)
        tokens_with_info = JavaTokenizer.tokenize(code)
        
        instances = []
        state = LineState()
        
        for token, token_type in tokens_with_info:
            if self._process_token(token, token_type, state, instances, file_id, max_examples, current_count):
                current_count += 1
                
        return instances, current_count >= max_examples

    def _process_token(self, token: str, token_type: TokenType, state: LineState, 
                      instances: List[CodeInstance], file_id: str, 
                      max_examples: int, current_count: int) -> bool:
        """Process a single token and return True if a new instance was created."""
        if token_type == TokenType.WHITESPACE:
            return self._handle_whitespace(token, state, instances, file_id, max_examples, current_count)
        
        state.add_token(token, token_type)
        state.update_depth(token)
        state.space_count = state.newline_count = 0
        return False

    def _handle_whitespace(self, token: str, state: LineState, instances: List[CodeInstance], 
                          file_id: str, max_examples: int, current_count: int) -> bool:
        """Handle whitespace token and create instance if needed."""
        state.update_whitespace(token)
        
        if state.newline_count > 0 and state.current_line:
            instance = self._create_instance(state, file_id)
            if instance and current_count < max_examples:
                instances.append(instance)
                state.reset(token.count(' ') + token.count('\t'))
                return True
                
        return False

    def _create_instance(self, state: LineState, file_id: str) -> Optional[CodeInstance]:
        """Create a new CodeInstance if line hasn't been seen before."""
        line_hash = hash(''.join(state.current_line).strip())
        if line_hash not in self.line_hashes:
            self.line_hashes.add(line_hash)
            return CodeInstance(
                file_id=file_id,
                tokens=state.tokens.copy(),
                token_types=[t.value for t in state.types],
                spaces=[min(s, self.max_spaces) for s in state.spaces],
                newlines=min(state.newline_count, self.max_newlines),
                scope_depth=state.depth
            )
        return None

    @staticmethod
    def _generate_file_id(code: str) -> str:
        """Generate a unique file ID from code content."""
        return hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
    

class DatasetManager:
    """Manages dataset creation, splitting, and saving operations."""
    
    def __init__(self, creator: DatasetCreator, max_examples: int, test_ratio: float):
        self.creator = creator
        self.max_examples = max_examples
        self.test_ratio = test_ratio
        self.instances_by_file: Dict[str, List[CodeInstance]] = {}
        self.current_count = 0

    def process_repository(self, repo_path: Path) -> None:
        """Process all Java files in the repository."""
        for java_file in repo_path.rglob('*.java'):
            if not self._process_file(java_file):
                break

    def save_datasets(self, train_path: Path, test_path: Path) -> None:
        """Split and save the datasets."""
        train_ids, test_ids = self._split_file_ids()
        self._save_split(train_path, train_ids)
        self._save_split(test_path, test_ids)

    def _process_file(self, java_file: Path) -> bool:
        """Process a single Java file. Returns False if max examples reached."""
        print(f'Processing {java_file.name}...')
        instances, done = self.creator.process_file(java_file, self.max_examples, self.current_count)
        
        for instance in instances:
            if instance.file_id not in self.instances_by_file:
                self.instances_by_file[instance.file_id] = []
            self.instances_by_file[instance.file_id].append(instance)
            self.current_count += 1
            
        return not done

    def _split_file_ids(self) -> Tuple[Set[str], Set[str]]:
        """Split file IDs into training and test sets."""
        file_ids = list(self.instances_by_file.keys())
        random.shuffle(file_ids)
        split_idx = int(len(file_ids) * (1 - self.test_ratio))
        return set(file_ids[:split_idx]), set(file_ids[split_idx:])

    def _save_split(self, path: Path, file_ids: Set[str]) -> None:
        """Save dataset split to file."""
        with jsonlines.open(path, mode='w') as writer:
            for file_id in file_ids:
                for instance in self.instances_by_file[file_id]:
                    writer.write(instance.__dict__)
