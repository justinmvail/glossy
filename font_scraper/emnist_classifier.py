"""
EMNIST Character Classifier

A lightweight CNN for single character recognition with confidence scores.
Trained on EMNIST ByClass dataset (62 classes: 0-9, A-Z, a-z).

Usage:
    classifier = EMNISTClassifier()
    classifier.load_model()  # Downloads/loads pre-trained weights

    # Classify a PIL image
    char, confidence, top3 = classifier.classify(image)
    print(f"Predicted: {char} ({confidence:.1%})")

    # Get detailed scores
    scores = classifier.get_scores(image)  # Dict[char, probability]
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

# Reverse mapping
CHAR_TO_IDX = {c: i for i, c in enumerate(EMNIST_CLASSES)}


class EMNISTNet(nn.Module):
    """Simple CNN for EMNIST classification."""

    def __init__(self, num_classes: int = 62):
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

    def forward(self, x):
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
    """
    EMNIST character classifier with confidence scores.

    Attributes:
        model: The CNN model
        device: CPU or CUDA device
        classes: List of 62 character classes
    """

    MODEL_PATH = Path(__file__).parent / "emnist_model.pt"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the classifier.

        Args:
            device: 'cuda', 'cpu', or None for auto-detect
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = None
        self.classes = EMNIST_CLASSES

    def load_model(self, model_path: Optional[Path] = None):
        """Load pre-trained model weights."""
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
        """Train the model on EMNIST ByClass dataset."""
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
        """
        Preprocess image for classification.

        Args:
            image: PIL Image (any size, RGB or grayscale)

        Returns:
            Tensor of shape (1, 1, 28, 28)
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

        # EMNIST images are transposed (rotated 90° + flipped)
        # We need to match that orientation
        img_array = np.transpose(img_array)
        img_array = np.flip(img_array, axis=0).copy()

        # To tensor
        tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def classify(self, image: Image.Image) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Classify a single character image.

        Args:
            image: PIL Image containing a single character

        Returns:
            Tuple of (predicted_char, confidence, top3_predictions)
            - predicted_char: The most likely character
            - confidence: Probability (0-1) of the prediction
            - top3_predictions: List of (char, probability) for top 3
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
        """
        Get probability scores for all 62 characters.

        Args:
            image: PIL Image containing a single character

        Returns:
            Dict mapping each character to its probability
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
        """
        Validate if a stroke image matches the expected character.

        Args:
            stroke_image: PIL Image of rendered strokes
            expected_char: The character that should be recognized

        Returns:
            Tuple of (passed, recognized, confidence, expected_probability)
            - passed: True if recognized matches expected
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


def test_classifier():
    """Test the classifier with some sample characters."""
    from PIL import ImageDraw, ImageFont

    print("Testing EMNIST Classifier\n")

    classifier = EMNISTClassifier()
    classifier.load_model()

    # Create test images
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 48)
    except:
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
