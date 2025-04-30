import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np

def validate_model(model, valloader, device='cpu', num_classes = 0, labels_name = None, show_cm = False):
    correct = 0
    total = 0
    model.eval()
    all_predicts = []
    all_labels = []

    with torch.no_grad():
        for (images, labels) in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

            if show_cm is True:
                all_predicts.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    print(f"Accuracy on {len(valloader) * valloader.batch_size} val images: {100*(correct/total)}%")

    if show_cm is True:
        show_confusion_matrix(np.arange(num_classes), labels_name, all_labels, all_predicts)
        print("\nRelatório de Classificação:\n", classification_report(all_labels, all_predicts))

    return correct, total

def show_confusion_matrix(labels, labels_name, expected, predicted):
    cm = confusion_matrix(y_true = expected, y_pred = predicted, labels = labels)
    if len(labels) > 50: 
        print(cm)
    else:
        cm_disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels_name)
        cm_disp.plot()
        plt.show()



























