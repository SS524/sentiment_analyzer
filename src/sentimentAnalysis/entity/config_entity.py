from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class DataProcessingConfig:
    root_dir: Path
    word2vec_modl_file: Path
    final_data_file: Path




@dataclass(frozen=True)
class ModelBuildingConfig:
    root_dir: Path
    base_modl_file: Path
    trained_modl_file: Path
    test_data_file: Path
    number_of_neurons_in_first_layer: int
    number_of_neurons_in_second_layer: int
    number_of_neurons_in_output_layer: int
    metrics: str
    learning_rate: float
    epochs: int




@dataclass(frozen=True)
class ModelEvaluationConfig:
    trained_model_path: Path
    test_data_path: Path





