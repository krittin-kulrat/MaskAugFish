import glob
import json
import time
import numpy as np
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    MulticlassAUROC,
)

from maskaugfish.dataloader import make_dataloaders
from maskaugfish.training import load_model


def evaluate(checkpoint_path, samples, class_to_id, batch_size, img_size, num_workers,regime, augmentation):
    num_class = len(class_to_id)
    id_to_class = {v: k for k, v in class_to_id.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    files = glob.glob(str(checkpoint_path))
    if len(files) == 0:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    elif len(files) != 5:
        raise ValueError(f"Multiple checkpoints found at {checkpoint_path}: {files}")

    models = []
    results = []
    for file in files:
        tmp = torch.load(file)
        model, img_size = load_model(tmp['info']['backbone'],
                                    num_class,
                                    device)
        model.load_state_dict(tmp['weight'])
        model.eval()
        models.append(model)




    # create test dataloader
    _, _, test_loader = make_dataloaders(samples=samples,
                                         batch_size=batch_size,
                                        num_workers=num_workers,
                                        img_size=img_size,
                                        weighted_train=False,
                                        seed=42,
                                        test_ratio=0.2,
                                        n_splits=5,
                                        fold_id=0,
                                        augment_pipeline=None
                                        )
    # metrics
    acc_metrics = [MulticlassAccuracy(num_classes=num_class, average="macro").to(device) for _ in range(5)]
    f1_metrics = [MulticlassF1Score(num_classes=num_class, average="macro").to(device) for _ in range(5)]
    cm_metrics = [MulticlassConfusionMatrix(num_classes=num_class).to(device) for _ in range(5)]
    auc_metrics = [MulticlassAUROC(num_classes=num_class, average="macro").to(device) for _ in range(5)]

    # inference

    total_time = 0.0
    total_images = 0
    for fold_idx, model in enumerate(models):
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
                acc_metrics[fold_idx](probs, y)
                f1_metrics[fold_idx](probs, y)
                cm_metrics[fold_idx](probs, y)
                auc_metrics[fold_idx](probs, y)

    # compute metrics
    for fold_idx in range(5):
        macro_acc = acc_metrics[fold_idx].compute().item()
        macro_f1 = f1_metrics[fold_idx].compute().item()
        cm = cm_metrics[fold_idx].compute().cpu().numpy()  # Convert to numpy for plotting
        macro_auc = auc_metrics[fold_idx].compute().item()
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
            "checkpoint": str(files[fold_idx]),
            "macro_accuracy": float(macro_acc),
            "macro_f1": float(macro_f1),
            "macro_auc": float(macro_auc),
            "confusion_matrix": cm.tolist(),
            "class_to_id": class_to_id,
            "regime": regime,
            "augmentation": augmentation,
        }

        result["per_class_metrics"] = {
            id_to_class[i]: {
                "support": int(support[i]),
                "accuracy": float(acc_per_class[i]),
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(num_class)
        }
        results.append(result)


    # # save confusion matrix plot
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
    #             xticklabels=[id_to_class[i] for i in range(num_class)],
    #             yticklabels=[id_to_class[i] for i in range(num_class)])
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.tight_layout()
    # cm_path = Path(checkpoint_path).with_suffix(".cm.png")
    # plt.savefig(cm_path)
    # plt.close()
    # print(f"Saved confusion matrix to {cm_path}")

    # # save ROC curves
    # plt.figure(figsize=(10, 8))
    # for c in range(num_classes):
    #     fpr, tpr, _ = roc_curve((all_labels == c).numpy(), all_probs[:, c].numpy())
    #     roc_auc = auc(fpr, tpr)
    #     class_name = id_to_class.get(c, f"Class {c}")
    #     plt.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")

    # plt.plot([0, 1], [0, 1], "k--", label="Random")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Multi-class ROC Curves")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # roc_path = Path(checkpoint_path).with_suffix(".roc.png")
    # plt.savefig(roc_path)
    # plt.close()
    # print(f"Saved ROC curves to {roc_path}")

        print(f"\n---- Evaluation Results ---- (Fold {fold_idx})")
        print(f"Macro Accuracy: {macro_acc:.4f}")
        print(f"Macro F1 Score: {macro_f1:.4f}")
        print(f"Macro AUC-ROC: {macro_auc:.4f}")
    # Save results next to checkpoint
    with open(checkpoint_path.replace("*.pth", "eval.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved evaluation results to {checkpoint_path.replace('*.pth', 'eval.json')}")

    return results

