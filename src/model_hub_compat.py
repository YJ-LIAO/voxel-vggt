try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    class PyTorchModelHubMixin:
        pass


try:
    from transformers.file_utils import ModelOutput
except ImportError:
    class ModelOutput:
        pass
