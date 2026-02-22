from unittest.mock import patch

from server_model import GPUInfo, infer_model_size_gb, select_gpus_for_model


def test_infer_model_size_from_name():
    assert infer_model_size_gb("mistral-7B") == 14.0
    assert infer_model_size_gb("Mixtral-8x7B") == 112.0
    assert infer_model_size_gb("unknown-model") is None


@patch("server_model.get_gpu_info")
def test_select_gpus_success(mock_gpu_info):
    mock_gpu_info.return_value = [
        GPUInfo(index=0, total_gb=48.0, free_gb=44.0),
        GPUInfo(index=1, total_gb=48.0, free_gb=43.0),
        GPUInfo(index=2, total_gb=48.0, free_gb=10.0),
    ]

    selected, status = select_gpus_for_model(
        model_name="mistral-7B",
        tp_size=2,
        gpu_util=0.9,
        allowed_gpu_ids=None,
    )

    assert status == "ok"
    assert selected == [0, 1]


@patch("server_model.get_gpu_info")
def test_select_gpus_no_gpus(mock_gpu_info):
    mock_gpu_info.return_value = []
    selected, status = select_gpus_for_model(
        model_name="mistral-7B", tp_size=1, gpu_util=0.8
    )

    assert selected is None
    assert status == "no gpus"


@patch("server_model.get_gpu_info")
def test_select_gpus_no_space(mock_gpu_info):
    mock_gpu_info.return_value = [
        GPUInfo(index=0, total_gb=48.0, free_gb=2.0),
        GPUInfo(index=1, total_gb=48.0, free_gb=2.5),
    ]

    selected, status = select_gpus_for_model(
        model_name="mistral-7B",
        tp_size=1,
        gpu_util=0.8,
    )

    assert selected is None
    assert status == "no space"


@patch("server_model.get_gpu_info")
def test_select_gpus_unknown_model_size(mock_gpu_info):
    mock_gpu_info.return_value = [GPUInfo(index=0, total_gb=48.0, free_gb=40.0)]

    selected, status = select_gpus_for_model(
        model_name="my-private-model",
        tp_size=1,
        gpu_util=0.8,
        model_size_gb=None,
    )

    assert selected is None
    assert status == "error loading model size"
