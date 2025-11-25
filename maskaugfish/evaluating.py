import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import time
import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from dataloader import make_dataloaders   
from training import load_model   

def evaluate(checkpoint_path, data_root, batch_size, img_size, num_workers):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint 
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    state_dict = ckpt["state_dict"]
    class_to_id = ckpt["class_to_id"]
    backbone = ckpt.get("backbone", "resnet18")
    
    num_classes = len(class_to_id)
    id_to_class = {v: k for k, v in class_to_id.items()}

    # load model 
    model, _ = load_model(backbone, num_classes, device)
    model.load_state_dict(state_dict)
    model.eval()

    # create test dataloader 
    _, _, test_loader = make_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        augment_pipeline=None,
        weighted_train=False,
        seed=42,
        test_ratio=0.2,
        n_splits=1,
        fold_id=0,
    )

    # metrics 
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
    auc_metric = MulticlassAUROC(num_classes=num_classes, average="macro").to(device)

    # inference 
    all_preds = []
    all_labels = []
    all_probs = []
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        if device.type == "cuda":
            dummy = torch.randn(1, 3, img_size, img_size).to(device)
            model(dummy)
            torch.cuda.synchronize()
        
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            logits = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_images += x.size(0)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # compute metrics 
    macro_acc = acc_metric(all_preds, all_labels).item()
    macro_f1 = f1_metric(all_preds, all_labels).item()
    cm = cm_metric(all_preds, all_labels).cpu().numpy()  # Convert to numpy for plotting
    macro_auc = auc_metric(all_probs, all_labels).item()
    avg_batch_time = total_time / len(test_loader)
    avg_image_time = total_time / total_images

    # per-class support (actual class count)
    support = cm.sum(axis=1)

    # per-class predictions (predicted count)
    pred_count = cm.sum(axis=0)

    # correct predictions per class
    correct = np.diag(cm)

    eps = 1e-8

    # per-class accuracy = correct / actual
    acc_per_class = correct / (support + eps)

    # per-class precision = correct / predicted
    precision_per_class = correct / (pred_count + eps)

    # per-class recall = correct / actual (same as accuracy)
    recall_per_class = correct / (support + eps)

    # per-class F1
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class + eps
    )


    # summary 
    result = {
        "checkpoint": str(checkpoint_path),
        "backbone": backbone,
        "num_classes": num_classes,
        "macro_accuracy": float(macro_acc),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        "confusion_matrix": cm.tolist(),
        "class_to_id": class_to_id,
        "inference_total_sec": total_time,
        "inference_per_batch_sec": avg_batch_time,
        "inference_per_image_ms": avg_image_time * 1000,
    }

    result["per_class_metrics"] = {
        id_to_class[i]: {
            "support": int(support[i]),
            "accuracy": float(acc_per_class[i]),
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
        }
        for i in range(num_classes)
    }

    # Save results next to checkpoint
    out_path = Path(checkpoint_path).with_suffix(".eval.json")
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved evaluation results to {out_path}")

    # save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", 
                xticklabels=[id_to_class[i] for i in range(num_classes)],
                yticklabels=[id_to_class[i] for i in range(num_classes)])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = Path(checkpoint_path).with_suffix(".cm.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # save ROC curves 
    plt.figure(figsize=(10, 8))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve((all_labels == c).numpy(), all_probs[:, c].numpy())
        roc_auc = auc(fpr, tpr)
        class_name = id_to_class.get(c, f"Class {c}")
        plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curves")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    roc_path = Path(checkpoint_path).with_suffix(".roc.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Saved ROC curves to {roc_path}")

    print("\n---- Evaluation Results ----")
    print(f"Macro Accuracy: {macro_acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Macro AUC-ROC: {macro_auc:.4f}")
    print(f"Inference per image: {avg_image_time*1000:.3f} ms")
    print(f"Inference per batch: {avg_batch_time:.6f} sec")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_root", required=True, help="Root directory containing the dataset")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--img_size", type=int, default=224, help="Image size for evaluation")
    parser.add_argument("--workers", type=int, default=0, help="Number of data loader workers")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        batch_size=args.batch,
        img_size=args.img_size,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
