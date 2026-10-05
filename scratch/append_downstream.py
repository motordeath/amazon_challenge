import sys
content = """

# Downstream
if len(new_true_positives) > 1000:
    print('\\n[6] Vector retrieval materially improved recall! Running downstream pipeline...')
    from business_entity_resolution.pipeline.labeling import label_candidates
    from business_entity_resolution.features.build_feature_matrix import build_feature_matrix
    from business_entity_resolution.models.train_lgbm import split_all_s1_ids, train_lgbm
    from business_entity_resolution.models.threshold import tune_global_threshold, predict_matches
    from business_entity_resolution.evaluation.report import full_validation_report

    # Step 4: Labeling
    print('\\n[7] Labeling Candidate Pairs...')
    labeled = label_candidates(union_capped, gt_long)
    del union_capped
    gc.collect()

    y = labeled['label'].to_numpy()
    small_labeled = labeled[['source1_entity_id', 'candidate_entity_id', 'score']]
    del labeled
    gc.collect()

    # Step 5: Features
    print('\\n[8] Building Feature Matrix...')
    others_n = pd.concat([s2_subset, s3_subset], ignore_index=True)
    del s2_subset, s3_subset
    gc.collect()

    X, meta = build_feature_matrix(small_labeled, s1n, others_n, cfg.features, chunk_size=4_000_000)
    del others_n, small_labeled
    gc.collect()

    # Step 6: LightGBM Training
    print('\\n[9] Training LightGBM...')
    groups = meta['source1_entity_id']
    train_ids, val_ids = split_all_s1_ids(s1n['entity_id'], n_splits=5, fold=0, seed=cfg.random_seed)
    train_mask = groups.isin(train_ids).to_numpy()
    val_mask = groups.isin(val_ids).to_numpy()

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    meta_val = meta[val_mask].reset_index(drop=True)
    del X, y
    gc.collect()

    model = train_lgbm(X_train, y_train, X_val, y_val, cfg.model)

    # Step 7: Threshold Tuning & Validation
    print('\\n[10] Threshold Tuning & Final Validation...')
    val_s1_ids = sorted(val_ids)
    gt_val = gt_long[gt_long['source1_entity_id'].isin(val_ids)]
    gt_val_dict = gt_val.groupby('source1_entity_id')['matched_entity_id'].apply(set).to_dict()
    gt_val_dict = {sid: gt_val_dict.get(sid, set()) for sid in val_s1_ids}

    proba_val = model.predict_proba(X_val)[:, 1]
    scored_val = meta_val.copy()
    scored_val['proba'] = proba_val

    thresh_report = tune_global_threshold(scored_val, gt_val_dict, val_s1_ids, cfg.threshold)
    report['threshold'] = thresh_report['best']
    tau = thresh_report['best']['tau']

    val_predictions = predict_matches(scored_val, tau, apply_reverse_constraint=True)
    full_report = full_validation_report(val_predictions, gt_val_dict, beta=cfg.threshold.beta)
    report['validation'] = full_report

    print(f"\\nDownstream F0.5: {full_report['macro_f_beta']:.4f}")

    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'\\nSaved downstream report to {out_path}')
"""
with open(r'D:\ml_amazon\kaggle_upload_src\code\business_entity_resolution\exp3_script.py', 'a') as f:
    f.write(content)
