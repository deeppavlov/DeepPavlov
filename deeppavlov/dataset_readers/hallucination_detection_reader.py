import json
import random
import re
from logging import getLogger
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_reader import DatasetReader
from deeppavlov.core.data.utils import download_decompress

log = getLogger(__name__)

@register('hallucination_reader')  
class HallucinationDatasetReader(DatasetReader):
    """Dataset reader for hallucination detection in RAG systems (DeepPavlov NER format)"""
    
    def read(self,
             data_path: str,
             dataset_name: str = None,
             language: str = "en",
             download_url: str = None,
             preprocess_raw: bool = False,
             raw_data_files: Optional[Dict[str, str]] = None,
             save_preprocessed: bool = True,
             validation_split: float = 0.1,
             random_seed: int = 42,
             create_validation_from_train: bool = True,
             *args, **kwargs) -> Dict[str, List[Tuple[List[str], List[str]]]]:
        """
        Read hallucination detection dataset in DeepPavlov NER format
        
        Returns:
            Dictionary with train/dev/test splits containing (tokens, bio_labels) tuples
        """
        
        self.language = language
        self.dataset_name = dataset_name or "ragtruth"
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.create_validation_from_train = create_validation_from_train
        
        # Set random seed for reproducible splits
        random.seed(self.random_seed)
        
        data_path = Path(data_path)
        
        # Check if we need to preprocess raw data
        if preprocess_raw or self._should_preprocess_raw_data(data_path, raw_data_files):
            log.info("Preprocessing raw RAGTruth data...")
            all_samples = self._preprocess_raw_ragtruth_data(data_path, raw_data_files)
            
            if save_preprocessed:
                self._save_preprocessed_data(data_path, all_samples)
        else:
            # Load existing preprocessed data
            all_samples = self._load_existing_data(data_path, download_url)
        
        # Filter by language if specified
        if self.language:
            all_samples = [s for s in all_samples if s.get('language', 'en') == self.language]
        
        # Convert to DeepPavlov format and split by train/dev/test
        dataset = {}
        splits_info = {}
        
        for split in ['train', 'dev', 'test']:
            split_samples = [s for s in all_samples if s['split'] == split]
            # dataset[split] = split_samples
            #TODO
            #HOTFIX
            dataset[split] = self._convert_to_xy_tuples(split_samples)[:1000]
            splits_info[split] = len(dataset[split])
            log.info(f"Loaded {len(dataset[split])} samples for {split} split")
        
        dataset = self._handle_validation_split(dataset, splits_info)
        
        return dataset
    
    def _convert_to_xy_tuples(self, samples: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Convert samples to (x, y) tuples format like CoNLL reader
        
        x будет содержать входные данные (prompt, answer)
        y будет содержать метки (labels)
        """
        xy_tuples = []
        
        for sample in samples:
            try:
                x = {
                    'prompt': sample.get('prompt', ''),
                    'answer': sample.get('answer', ''),
                    'task_type': sample.get('task_type', ''),
                    'dataset': sample.get('dataset', ''),
                    'language': sample.get('language', ''),
                    'labels': sample.get('labels', []),
                }
                
                y = {
                    'labels': sample.get('labels', [])
                }
                
                xy_tuples.append((x, y))
                
            except Exception as e:
                log.warning(f"Failed to process sample: {e}")
                continue
        
        return xy_tuples
    def _should_preprocess_raw_data(self, data_path: Path, raw_data_files: Optional[Dict[str, str]]) -> bool:
        """Check if we should preprocess raw data"""
        if raw_data_files:
            return True
            
        # Check for raw RAGTruth files
        response_file = data_path / "response.jsonl"
        source_file = data_path / "source_info.jsonl"
        
        if response_file.exists() and source_file.exists():
            # Check if preprocessed file already exists
            preprocessed_file = data_path / f"{self.dataset_name}_data.json"
            return not preprocessed_file.exists()
        
        return False
    
    def _preprocess_raw_ragtruth_data(self, 
                                    data_path: Path, 
                                    raw_data_files: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Preprocess raw RAGTruth data"""
        
        # Determine file paths
        if raw_data_files:
            response_file = Path(raw_data_files.get('response_file', ''))
            source_file = Path(raw_data_files.get('source_file', ''))
        else:
            response_file = data_path / "response.jsonl"
            source_file = data_path / "source_info.jsonl"
        
        if not response_file.exists() or not source_file.exists():
            raise FileNotFoundError(
                f"Raw data files not found: {response_file}, {source_file}"
            )
        
        log.info(f"Loading raw data from {response_file} and {source_file}")
        
        # Load raw data
        responses, sources = self._load_raw_data(response_file, source_file)
        
        # Create index of sources by ID
        sources_by_id = {source["source_id"]: source for source in sources}
        
        # Process each response
        processed_samples = []
        
        for response in responses:
            try:
                source_id = response["source_id"]
                if source_id not in sources_by_id:
                    log.warning(f"Source ID {source_id} not found in sources")
                    continue
                
                source = sources_by_id[source_id]
                sample = self._create_sample_from_raw(response, source)
                processed_samples.append(sample)
                
            except Exception as e:
                log.warning(f"Failed to process response {response.get('source_id', 'unknown')}: {e}")
                continue
        
        log.info(f"Preprocessed {len(processed_samples)} samples from raw data")
        return processed_samples
    
    def _load_raw_data(self, response_file: Path, source_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """Load raw RAGTruth data files"""
        
        # Load responses
        responses = []
        with open(response_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    response = json.loads(line.strip())
                    responses.append(response)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse response line {line_num}: {e}")
        
        # Load sources
        sources = []
        with open(source_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    source = json.loads(line.strip())
                    sources.append(source)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse source line {line_num}: {e}")
        
        log.info(f"Loaded {len(responses)} responses and {len(sources)} sources")
        # return responses[:200], sources[:200]        
        return responses, sources
    
    def _create_sample_from_raw(self, response: Dict, source: Dict) -> Dict[str, Any]:
        """Create sample dict from raw response and source data"""
        
        prompt = source["prompt"]
        answer = response["response"]
        split = response["split"]
        task_type = source["task_type"]
        
        # Process labels
        labels = []
        for label in response.get("labels", []):
            start_char = label["start"]
            end_char = label["end"]
            labels.append({
                "start": start_char,
                "end": end_char,
                "label": label.get("label_type", "hallucination"),
            })
        
        return {
            "prompt": prompt,
            "answer": answer,
            "labels": labels,
            "split": split,
            "task_type": task_type,
            "dataset": self.dataset_name,
            "language": self.language,
        }
    
    def _save_preprocessed_data(self, data_path: Path, samples: List[Dict[str, Any]]):
        """Save preprocessed data to JSON file"""
        output_file = data_path / f"{self.dataset_name}_data.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved preprocessed data to {output_file}")
    
    def _load_existing_data(self, data_path: Path, download_url: str = None) -> List[Dict[str, Any]]:
        """Load existing preprocessed data"""
        
        # Check for preprocessed files
        json_files = list(data_path.glob(f'{self.dataset_name}*.json')) + list(data_path.glob('*.json'))
        
        if not json_files and download_url:
            log.info(f"Dataset not found in {data_path}, downloading from {download_url}")
            data_path.mkdir(exist_ok=True, parents=True)
            download_decompress(download_url, data_path)
            json_files = list(data_path.glob(f'{self.dataset_name}*.json')) + list(data_path.glob('*.json'))
        
        if not json_files:
            raise RuntimeError(f'No JSON files found in "{data_path}"')
        
        # Load and parse dataset
        all_samples = []
        for file_path in json_files:
            if file_path.name.endswith('_data.json') or file_path.name == f'{self.dataset_name}.json':
                samples = self._load_samples_from_file(file_path)
                all_samples.extend(samples)
        
        return all_samples
    
    def _load_samples_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load samples from JSON or JSONL file"""
        samples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix == '.jsonl':
                # JSONL format
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            sample = json.loads(line)
                            samples.append(sample)
                        except json.JSONDecodeError as e:
                            log.warning(f"Failed to parse line in {file_path}: {e}")
            else:
                # JSON format
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    elif isinstance(data, dict):
                        if 'samples' in data:
                            samples.extend(data['samples'])
                        else:
                            samples.append(data)
                except json.JSONDecodeError as e:
                    log.error(f"Failed to parse JSON file {file_path}: {e}")
        
        return samples
    
    def _handle_validation_split(self, dataset: Dict[str, List], splits_info: Dict[str, int]) -> Dict[str, List]:
        """Handle validation split creation and renaming"""
        
        # Check existing splits
        has_dev = 'dev' in dataset and len(dataset['dev']) > 0
        has_valid = 'valid' in dataset and len(dataset['valid']) > 0
        has_train = 'train' in dataset and len(dataset['train']) > 0
        
        if has_dev and not has_valid:
            # Rename dev to valid
            dataset['valid'] = dataset['dev']
            del dataset['dev']
            log.info(f"Renamed 'dev' split to 'valid' ({len(dataset['valid'])} samples)")
            
        elif not has_dev and not has_valid and has_train and self.create_validation_from_train:
            # Create validation split from training data
            dataset = self._create_validation_from_train(dataset)
            
        elif not has_dev and not has_valid:
            log.warning("No validation data found and create_validation_from_train=False")
            dataset['valid'] = []
        
        return dataset
    
    def _create_validation_from_train(self, dataset: Dict[str, List]) -> Dict[str, List]:
        """Split training data into train and validation sets"""
        
        train_samples = dataset['train']
        
        if len(train_samples) == 0:
            log.warning("No training samples found, cannot create validation split")
            dataset['valid'] = []
            return dataset
        
        # Calculate split size
        val_size = int(len(train_samples) * self.validation_split)
        val_size = max(1, val_size)
        val_size = min(val_size, len(train_samples) - 1)
        
        # Shuffle and split
        shuffled_samples = train_samples.copy()
        random.shuffle(shuffled_samples)
        
        # Split the data
        train_split = shuffled_samples[val_size:]
        valid_split = shuffled_samples[:val_size]
        
        dataset['train'] = train_split
        dataset['valid'] = valid_split
        
        log.info(f"Created validation split: train={len(train_split)}, valid={len(valid_split)} "
                f"(split ratio: {self.validation_split})")
        
        return dataset

@register('ragtruth_reader')
class RAGTruthDatasetReader(HallucinationDatasetReader):
    """Specialized reader for RAGTruth dataset"""
    
    def read(self, data_path: str, **kwargs) -> Dict[str, List[Tuple[List[str], List[str]]]]:
        """Read RAGTruth dataset with default settings"""
        
        defaults = {
            'dataset_name': 'ragtruth',
            'preprocess_raw': True,
            'language': 'en',
            'validation_split': 0.15,
            'create_validation_from_train': True,
        }
        
        config = {**defaults, **kwargs}
        return super().read(data_path, **config)