def split_classes(cfg):
    first_num_classes = cfg.first_num_classes
    learn_num_per_step = int(cfg.task.split('-')[1])
    cfg.extend = 0
    for i in range(cfg.step):
        first_num_classes += learn_num_per_step
        cfg.extend += learn_num_per_step

    total_number = cfg.total_num_classes - 1

    original = list(range(total_number + 1))
    to_learn = list(range(first_num_classes + 1))
    remaining = [i for i in original if i not in to_learn]
    if cfg.extend != 0:
        prefetch_cats = cfg.extend
        prefetch_cats = to_learn[-prefetch_cats:]
    else:
        prefetch_cats = to_learn
    # print(to_learn, prefetch_cats, remaining)
    # raise RuntimeError
    return to_learn, prefetch_cats, remaining