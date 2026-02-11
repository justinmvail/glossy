"""EMNIST Character Classifier.

A lightweight CNN for single character recognition with confidence scores,
trained on the EMNIST ByClass dataset (62 classes: 0-9, A-Z, a-z).

This module provides a pre-trained classifier for validating stroke-based
character generation. It can determine whether rendered strokes are
recognizable as the intended character.

Classes:
    EMNISTNet: Simple CNN architecture for 28x28 character images.
    EMNISTClassifier: High-level classifier API with preprocessing.

Usage:
    Basic classification::

        classifier = EMNISTClassifier()
        classifier.load_model()  # Downloads/loads pre-trained weights

        # Classify a PIL image
        char, confidence, top3 = classifier.classify(image)
        print(f"Predicted: {char} ({confidence:.1%})")

        # Get detailed scores
        scores = classifier.get_scores(image)  # Dict[char, probability]

    Stroke validation::

        passed, recognized, conf, expected_prob = classifier.validate_stroke(
            stroke_image, expected_char='A'
        )
        if passed:
            print(f"Stroke recognized correctly with {conf:.1%} confidence")

Attributes:
    EMNIST_CLASSES: String of all 62 character classes in order.
    CHAR_TO_IDX: Dict mapping characters to their class indices.

Note:
    The model expects images with white content on black background
    (EMNIST convention). The preprocess() method handles automatic
    inversion if needed.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# EMNIST ByClass has 62 classes
# Mapping from class index to character
EMNIST_CLASSES = (
    '0123456789'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'abcdefghijklmnopqrstuvwxyz'
)
"""str: All 62 EMNIST ByClass character classes.

The string is ordered by class index: digits 0-9 (indices 0-9),
uppercase A-Z (indices 10-35), lowercase a-z (indices 36-61).
"""

# Reverse mapping
CHAR_TO_IDX = {c: i for i, c in enumerate(EMNIST_CLASSES)}
"""dict: Mapping from character to EMNIST class index.

Example:
    >>> CHAR_TO_IDX['A']
    10
    >>> CHAR_TO_IDX['a']
    36
"""


class EMNISTNet(nn.Module):
    """Simple CNN for EMNIST classification.

    A 3-layer convolutional network with max pooling and dropout,
    designed for 28x28 grayscale character images.

    Architecture:
        - Conv1: 1 -> 32 channels, 3x3, padding 1, then MaxPool 2x2
        - Conv2: 32 -> 64 channels, 3x3, padding 1, then MaxPool 2x2
        - Conv3: 64 -> 128 channels, 3x3, padding 1, then MaxPool 2x2
        - FC1: 128*3*3 -> 256
        - FC2: 256 -> 62 (num_classes)

    Attributes:
        conv1, conv2, conv3: Convolutional layers.
        pool: Max pooling layer (2x2).
        dropout1, dropout2: Dropout layers (0.25 and 0.5).
        fc1, fc2: Fully connected layers.
    """

    def __init__(self, num_classes: int = 62):
        """Initialize the network.

        Args:
            num_classes: Number of output classes. Default 62 for EMNIST ByClass.
        """
        super().__init__()
        # Input: 1x28x28
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # After 3 pools: 28 -> 14 -> 7 -> 3
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        # Conv block 3
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        # Flatten and FC
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class EMNISTClassifier:
    """EMNIST character classifier with confidence scores.

    Provides a high-level API for character classification including
    image preprocessing, model loading, and result interpretation.

    Attributes:
        model: The underlying EMNISTNet CNN model.
        device: PyTorch device (CPU or CUDA).
        classes: String of all 62 character classes.
        MODEL_PATH: Default path to pre-trained model weights.

    Example:
        >>> classifier = EMNISTClassifier()
        >>> classifier.load_model()
        >>> char, conf, top3 = classifier.classify(some_image)
        >>> print(f"Best guess: {char} at {conf:.1%}")
    """

    MODEL_PATH = Path(__file__).parent / "emnist_model.pt"
    """Path: Default location for pre-trained model weights."""

    def __init__(self, device: Optional[str] = None):
        """Initialize the classifier.

        Args:
            device: PyTorch device string ('cuda', 'cpu'), or None
                for automatic detection based on CUDA availability.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = None
        self.classes = EMNIST_CLASSES

    def load_model(self, model_path: Optional[Path] = None):
        """Load pre-trained model weights.

        If weights file doesn't exist, trains a new model on EMNIST
        and saves the weights for future use.

        Args:
            model_path: Path to model weights file. Defaults to
                MODEL_PATH (emnist_model.pt in same directory).
        """
        if model_path is None:
            model_path = self.MODEL_PATH

        self.model = EMNISTNet(num_classes=62)

        if model_path.exists():
            print(f"Loading EMNIST model from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"No pre-trained model found at {model_path}")
            print("Training new model on EMNIST dataset...")
            self._train_model()
            # Save for next time
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

        self.model.to(self.device)
        self.model.eval()

    def _train_model(self, epochs: int = 5, batch_size: int = 128):
        """Train the model on EMNIST ByClass dataset.

        Downloads the EMNIST dataset if needed and trains for the
        specified number of epochs.

        Args:
            epochs: Number of training epochs.
            batch_size: Batch size for training.
        """
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        print("Downloading EMNIST dataset (this may take a few minutes)...")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Download EMNIST ByClass
        train_dataset = datasets.EMNIST(
            root='/tmp/emnist_data',
            split='byclass',
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.EMNIST(
            root='/tmp/emnist_data',
            split='byclass',
            train=False,
            download=True,
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

                if batch_idx % 500 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.1f}%")

            # Test accuracy
            self.model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)

            print(f"Epoch {epoch+1}/{epochs}: Train Acc: {100.*correct/total:.1f}%, "
                  f"Test Acc: {100.*test_correct/test_total:.1f}%")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for classification.

        Handles grayscale conversion, resizing to 28x28, automatic
        inversion (EMNIST expects white-on-black), normalization,
        and EMNIST orientation correction (transpose + flip).

        Args:
            image: PIL Image of any size, RGB or grayscale.

        Returns:
            Tensor of shape (1, 1, 28, 28) ready for model input.
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')

        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # EMNIST images have white text on black background
        # If our image is black text on white, invert it
        img_array = np.array(image)

        # Check if image is mostly white (black text on white bg)
        if img_array.mean() > 127:
            img_array = 255 - img_array

        # Normalize
        img_array = img_array.astype(np.float32) / 255.0
        img_array = (img_array - 0.1307) / 0.3081

        # EMNIST images are transposed (rotated 90 + flipped)
        # We need to match that orientation
        img_array = np.transpose(img_array)
        img_array = np.flip(img_array, axis=0).copy()

        # To tensor
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Classify a single character image.

        Args:
            image: PIL Image containing a single character.

        Returns:
            Tuple of (predicted_char, confidence, top3_predictions):
                - predicted_char: The most likely character
                - confidence: Probability (0-1) of the prediction
                - top3_predictions: List of (char, probability) for top 3

        Raises:
            RuntimeError: If model not loaded (call load_model() first).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        # Get top predictions
        top_probs, top_indices = probs.topk(3)

        predicted_idx = top_indices[0].item()
        predicted_char = self.classes[predicted_idx]
        confidence = top_probs[0].item()

        top3 = [(self.classes[idx.item()], prob.item())
                for idx, prob in zip(top_indices, top_probs)]

        return predicted_char, confidence, top3

    def get_scores(self, image: Image.Image) -> Dict[str, float]:
        """Get probability scores for all 62 characters.

        Args:
            image: PIL Image containing a single character.

        Returns:
            Dict mapping each character to its probability (0-1).

        Raises:
            RuntimeError: If model not loaded (call load_model() first).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        tensor = self.preprocess(image)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]

        return {char: probs[i].item() for i, char in enumerate(self.classes)}

    def validate_stroke(
        self,
        stroke_image: Image.Image,
        expected_char: str
    ) -> Tuple[bool, str, float, float]:
        """Validate if a stroke image matches the expected character.

        Useful for quality checking generated stroke data by verifying
        that the rendered strokes are recognizable as the intended character.

        Args:
            stroke_image: PIL Image of rendered strokes.
            expected_char: The character that should be recognized.

        Returns:
            Tuple of (passed, recognized, confidence, expected_probability):
                - passed: True if recognized matches expected (case-insensitive)
                - recognized: The predicted character
                - confidence: Probability of the prediction
                - expected_probability: Probability assigned to expected char
        """
        scores = self.get_scores(stroke_image)

        # Get prediction
        recognized = max(scores, key=scores.get)
        confidence = scores[recognized]

        # Get probability of expected character
        expected_upper = expected_char.upper() if len(expected_char) == 1 else expected_char
        expected_lower = expected_char.lower() if len(expected_char) == 1 else expected_char

        expected_prob = max(
            scores.get(expected_char, 0),
            scores.get(expected_upper, 0),
            scores.get(expected_lower, 0)
        )

        # Check if match (case-insensitive for letters)
        passed = (recognized.lower() == expected_char.lower())

        return passed, recognized, confidence, expected_prob


def test_classifier() -> None:
    """Test the classifier with sample rendered characters.

    Creates test images using a system font and runs classification
    on each, displaying results in a table format.

    Usage:
        python emnist_classifier.py
    """
    from PIL import ImageDraw, ImageFont

    print("Testing EMNIST Classifier\n")

    classifier = EMNISTClassifier()
    classifier.load_model()

    # Create test images
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    except OSError:
        font = ImageFont.load_default()

    test_chars = 'ABCabc012'

    print(f"{'Char':<6} {'Predicted':<10} {'Confidence':<12} {'Top 3'}")
    print("-" * 60)

    for char in test_chars:
        # Create image
        img = Image.new('L', (64, 64), 255)
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), char, font=font)
        x = (64 - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (64 - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)

        # Classify
        predicted, confidence, top3 = classifier.classify(img)

        status = '✓' if predicted == char or predicted.lower() == char.lower() else '✗'
        top3_str = ', '.join([f"{c}:{p:.0%}" for c, p in top3])

        print(f"{char:<6} {predicted:<10} {confidence:<12.1%} {top3_str} {status}")


if __name__ == '__main__':
    test_classifier()
