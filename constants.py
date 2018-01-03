train_offline_data_path = "resource/tb01ccf_offline_stage1_train.csv"
train_online_data_path = "resource/tb02ccf_online_stage1_train.csv"
test_offline_data_path = "resource/tb03ccf_offline_stage1_test_revised.csv"

# 特征值路径
train_off_user_feature_path = "resource/features/offline/trainUserFeature"
train_off_mer_feature_path = 'resource/features/offline/trainMerFeature'
train_off_user_mer_feature_path = "resource/features/offline/trainUserMerFeature"

train_online_user_feature_path = "resource/features/online/trainUserFeature"
train_online_mer_feature_path = "resource/features/online/trainMerFeature"
train_online_user_mer_feature_path = "resource/features/online/trainUserMerFeature"

model_path = "model/svm"
# 带有标签 0 1 的offline 训练数据
label_data_path = 'resource/trainlabeldata/train_label_data.csv'
# 未进行填充的feature

train_feature_path = 'resource/trainfeatures/train_features.csv'
# 仅仅通过0 和-1 进行填充缺失值
train_feature_filna_path ='resource/trainfeatures/train_features_filna.csv'

# 200000条数据
less_label_data='test/features/label_data.csv'
less_feature_data='test/features/feature.csv'

# 3000条数据
less_train_label = 'test/train/train_label.csv'
less_train_feature = 'test/train/train_features.csv'