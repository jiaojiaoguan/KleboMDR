#coding=utf-8
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.svm import LinearSVC

def final_SVM_model(data_type):
    """
    此函数使用嵌套 CV 进行无偏特征选择、K 优化和评估。
    - 外层 CV: StratifiedKFold (n_splits=3) 用于最终无偏评估。
    - 内层: LOO 用于 per-outer-train 的特征选/ K 优化。
    - Per-fold 预处理避免泄露。
    - 限制 K <=6，选 max AUROC 的 K。
    - 计算 overfit gap 监控。
    - 最终模型用全数据 fit（有 bias），但 metrics 来自外层 test。
    """

    # 读取标签文件和特征文件（同 select_topk）
    label_df = pd.read_csv('sample_label_0112.csv', sep=',', index_col=0)
    feature_df = pd.read_csv(f'{data_type}_tpm.tsv', sep='\t', index_col=0)

    common_samples = label_df.index.intersection(feature_df.index)
    label_df = label_df.loc[common_samples]
    feature_df = feature_df.loc[common_samples]

    result_summary = defaultdict(list)

    for drug in label_df.columns:
        print(f"\n==== Processing drug: {drug} ====")

        y = label_df[drug]
        valid_idx = y[y.isin([0, 1])].index

        if len(valid_idx) < 10:
            print(f"Skipping {drug}: too few valid samples ({len(valid_idx)})")
            continue

        X_valid = feature_df.loc[valid_idx].copy()
        y_valid = y.loc[valid_idx].copy()

        if y_valid.nunique() < 2:
            print(f"Skipping {drug}: only one class")
            continue

        print(f"Valid samples: {len(y_valid)}, Positive: {sum(y_valid == 1)}, Negative: {sum(y_valid == 0)}")

        # 嵌套 CV: 外层 for 无偏评估
        outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        outer_auc_list = []
        outer_acc_list = []
        outer_precision_list = []
        outer_recall_list = []
        outer_f1_list = []
        best_features_per_fold = []
        gaps = []  # overfit gaps

        for outer_train_idx, outer_test_idx in outer_cv.split(X_valid, y_valid):
            X_outer_train, X_outer_test = X_valid.iloc[outer_train_idx], X_valid.iloc[outer_test_idx]
            y_outer_train, y_outer_test = y_valid.iloc[outer_train_idx], y_valid.iloc[outer_test_idx]

            # 内层: 用 outer_train 做特征选/排序/K 优化（类似 select_topk 逻辑）
            # 先 log1p (非参数，低风险)
            X_outer_train_log = np.log1p(X_outer_train)
            X_outer_test_log = np.log1p(X_outer_test)

            # 内层标准化: 只 fit outer_train
            scaler = StandardScaler()
            X_outer_train_scaled = pd.DataFrame(scaler.fit_transform(X_outer_train_log),
                                                columns=X_outer_train.columns,
                                                index=X_outer_train.index)
            X_outer_test_scaled = pd.DataFrame(scaler.transform(X_outer_test_log),
                                               columns=X_outer_test.columns,
                                               index=X_outer_test.index)

            # 内层特征选择: LinearSVC on outer_train_scaled
            clf_inner = LinearSVC(
                penalty='l1', dual=False, loss='squared_hinge',
                class_weight='balanced', random_state=42, max_iter=1000,C=0.05,
            )
            clf_inner.fit(X_outer_train_scaled, y_outer_train)

            coef = clf_inner.coef_.ravel()
            nonzero_idx = np.where(coef != 0)[0]
            selected_features = X_outer_train_scaled.columns[nonzero_idx].tolist()
            selected_coefs = coef[nonzero_idx]

            if len(selected_features) < 1:
                print(f"Warning: Few features in inner fold for {drug}.")
                continue

            sorted_idx = np.argsort(np.abs(selected_coefs))[::-1]
            important_features_inner = [selected_features[i] for i in sorted_idx]

            # 内层 LOO on outer_train to select best K (<=6)
            loo = LeaveOneOut()
            max_features_inner = len(important_features_inner)
            k_values_inner = list(range(1, min(7, max_features_inner + 1), 1))  # 限 <=6, 步长1

            avg_auc_list_inner = []

            for k in k_values_inner:
                top_k_features = important_features_inner[:k]

                y_true_inner = []
                y_prob_inner = []

                for train_idx, test_idx in loo.split(X_outer_train_scaled):
                    X_train = X_outer_train_scaled.iloc[train_idx][top_k_features]
                    X_test = X_outer_train_scaled.iloc[test_idx][top_k_features]
                    y_train, y_test = y_outer_train.iloc[train_idx], y_outer_train.iloc[test_idx]

                    clf = SVC(kernel='linear', probability=True, class_weight='balanced',
                              random_state=42, max_iter=1000)
                    clf.fit(X_train, y_train)

                    y_prob = clf.predict_proba(X_test)[:, 1]
                    y_true_inner.append(y_test.values[0])
                    y_prob_inner.append(y_prob[0])

                auc = roc_auc_score(y_true_inner, y_prob_inner)
                avg_auc_list_inner.append(auc)

            # 内层选 best K: max AUROC in <=6
            max_idx = np.argmax(avg_auc_list_inner)
            best_k_inner = k_values_inner[max_idx]

            # 用内层 best_k_inner 特征评估 outer_test
            final_features_inner = important_features_inner[:best_k_inner]
            clf_temp = SVC(kernel='linear', probability=True, class_weight='balanced',
                           random_state=42, max_iter=1000)
            clf_temp.fit(X_outer_train_scaled[final_features_inner], y_outer_train)

            # 计算 overfit gap (inner train vs. outer test)
            y_prob_train = clf_temp.predict_proba(X_outer_train_scaled[final_features_inner])[:, 1]
            train_auc = roc_auc_score(y_outer_train, y_prob_train)

            y_prob_test = clf_temp.predict_proba(X_outer_test_scaled[final_features_inner])[:, 1]
            test_auc = roc_auc_score(y_outer_test, y_prob_test)
            gap = train_auc - test_auc
            gaps.append(gap)
            print(f"train_auc: {train_auc:.4f}, test_auc: {test_auc:.4f}, Inner fold gap: {gap:.4f}")

            y_pred_test = (y_prob_test > 0.5).astype(int)
            outer_auc_list.append(test_auc)
            outer_acc_list.append(accuracy_score(y_outer_test, y_pred_test))
            outer_precision_list.append(precision_score(y_outer_test, y_pred_test))
            outer_recall_list.append(recall_score(y_outer_test, y_pred_test))
            outer_f1_list.append(f1_score(y_outer_test, y_pred_test))

            best_features_per_fold.append(final_features_inner)

        # 汇总无偏性能（平均外层）
        unbiased_loo_auc = np.mean(outer_auc_list)
        unbiased_loo_acc = np.mean(outer_acc_list)
        unbiased_loo_precision = np.mean(outer_precision_list)
        unbiased_loo_recall = np.mean(outer_recall_list)
        unbiased_loo_f1 = np.mean(outer_f1_list)
        avg_gap = np.mean(gaps)

        print(f"Unbiased Nested CV AUC: {unbiased_loo_auc:.4f}")
        print(f"Unbiased Nested CV ACC: {unbiased_loo_acc:.4f}")
        print(f"Unbiased Nested CV Precision: {unbiased_loo_precision:.4f}")
        print(f"Unbiased Nested CV Recall: {unbiased_loo_recall:.4f}")
        print(f"Unbiased Nested CV F1: {unbiased_loo_f1:.4f}")
        print(f"Average overfit gap: {avg_gap:.4f}")

        # 最终特征：取所有 fold 最频繁特征（限 6 个）
        from collections import Counter

        for fold_feats in best_features_per_fold:
            print(fold_feats)

        all_features = [f for fold_feats in best_features_per_fold for f in fold_feats]
        final_features = [feat for feat, _ in Counter(all_features).most_common(6)]  # top-6

        print(f"Final selected features: {final_features}")

        # 训练最终模型（用全数据，作为生产模型；有 bias，但用于部署）
        X_final_log = np.log1p(X_valid)
        scaler_final = StandardScaler()
        X_final_scaled = scaler_final.fit_transform(X_final_log)  # full fit，有 bias
        X_final = pd.DataFrame(X_final_scaled, columns=X_valid.columns, index=X_valid.index)[final_features]

        clf_final = SVC(
            kernel='linear', probability=True, class_weight='balanced',
            random_state=42, max_iter=1000
        )
        clf_final.fit(X_final, y_valid)

        # 保存最终模型
        joblib.dump(clf_final, f"SVM_{data_type}_step1/{drug}_{data_type}_final_model.pkl")

        # 记录结果
        result_summary['drug'].append(drug)
        result_summary['unbiased_loo_acc'].append(unbiased_loo_acc)
        result_summary['unbiased_loo_auc'].append(unbiased_loo_auc)
        result_summary['unbiased_loo_precision'].append(unbiased_loo_precision)
        result_summary['unbiased_loo_recall'].append(unbiased_loo_recall)
        result_summary['unbiased_loo_f1'].append(unbiased_loo_f1)
        result_summary['avg_gap'].append(avg_gap)
        result_summary['final_features'].append(final_features)

    # 输出汇总
    result_df = pd.DataFrame(result_summary)
    result_df.to_csv(f'SVM_{data_type}_step1/SVM_feature_selection_summary_0722_1.csv', index=False)
    print("\nSummary saved to 'SVM_feature_selection_summary.csv'")


if __name__ == '__main__':

    # final_SVM_model(data_type="WGS")
    final_SVM_model(data_type="NGS")