from maskaugfish.optimize import eval_with_added_aug


aug_list = [
    "channel_switch",
    "addition",
    "guassian_noise",
    "dropout",
    "gaussian_blur",
    "solarize",
    "equalize"
]

best_score = 0.0
best_params = None
current_augs = []
for step in range(len(aug_list)):
    best_step_score = 0.0
    best_step_params = None
    best_added_aug = None
    for i in range(len(aug_list)):
        if aug_list[i] in current_augs:
            continue
        print(f"Evaluating addition of augmentation: {aug_list[i]}")
        params, score = eval_with_added_aug(aug_list[i], current_augs)
        if score > best_step_score:
            best_step_score = score
            best_step_params = params
            best_added_aug = aug_list[i]

    if best_step_score > best_score:
        best_score = best_step_score
        best_params = best_step_params
        current_augs.append(best_added_aug)
        print(f"Added augmentation: {best_added_aug} with score: {best_step_score}")
        print(f"Current augmentations: {current_augs}")
        print(f"Best parameters so far: {best_params}")
    else:
        print("No further improvement, stopping augmentation addition.")
        break

print("Optimization completed.")
print(f"Final augmentations: {current_augs}")
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
