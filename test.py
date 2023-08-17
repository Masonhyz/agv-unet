import torch
import pytest
from utils import iou


def test_iou_with_empty_masks():
    y_true = torch.zeros((3, 4, 4))
    y_pred = torch.zeros((3, 4, 4))
    assert iou(y_true, y_pred).item() == pytest.approx(1.0, abs=1e-6)

def test_iou_with_perfect_overlap():
    y_true = torch.ones((3, 4, 4))
    y_pred = torch.ones((3, 4, 4))
    assert iou(y_true, y_pred).item() == pytest.approx(1.0, abs=1e-6)

def test_iou_with_no_overlap():
    y_true = torch.ones((3, 4, 4))
    y_pred = torch.zeros((3, 4, 4))
    assert iou(y_true, y_pred).item() == pytest.approx(0.0, abs=1e-6)

def test_iou_with_partial_overlap():
    y_true = torch.tensor([
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ])
    y_pred = torch.tensor([
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ])
    expected_iou = 0.5
    assert iou(y_true, y_pred).item() == pytest.approx(expected_iou, abs=1e-6)


def test_iou_with_partial_overlap2():
    y_true = torch.tensor([
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ])
    y_pred = torch.tensor([
        [[1, 0, 1, 1],
         [1, 1, 1, 1],
         [0, 1, 0, 0],
         [0, 0, 0, 0]],
        [[1, 1, 1, 1],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
    ])
    expected_iou = 9/24
    assert iou(y_true, y_pred).item() == pytest.approx(expected_iou, abs=1e-6)


def test_iou_with_different_shapes():
    y_true = torch.ones((3, 4, 4))
    y_pred = torch.zeros((3, 5, 5))
    with pytest.raises(RuntimeError):
        iou(y_true, y_pred)


if __name__ == "__main__":
    test_iou_with_partial_overlap2()
