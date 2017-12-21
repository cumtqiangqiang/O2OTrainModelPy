TRAIN_OFFLINE_DATA_PATH = "Resource/tb01ccf_offline_stage1_train.csv"
TRAIN_ONLINE_DATA_PATH = "Resource/tb02ccf_online_stage1_train.csv"
TEST_OFFLINE_DATA_PATH = "Resource/tb03ccf_offline_stage1_test_revised.csv"

# 特征值路径
TRAIN_OFF_USER_FEATURE_PATH = "Resource/features/offline/trainUserFeature"
TRAIN_OFF_MER_FEATURE_PATH = 'Resource/features/offline/trainMerFeature/part-00000.csv'
TRAIN_OFF_USER_MER_FEATURE_PATH = "Resource/features/offline/trainUserMerFeature"

TRAIN_ONLINE_USER_FEATURE_PATH = "Resource/features/online/trainUserFeature"
TRAIN_ONLINE_MER_FEATURE_PATH = "Resource/features/online/trainMerFeature"
TRAIN_ONLINE_USER_MER_FEATURE_PATH = "Resource/features/online/trainUserMerFeature"

# 带有标签 0 1 的offline 训练数据
label_data_path = 'Resource/trainLabelData/train_label_data.csv'

# 未进行填充的feature
train_feature_path = 'Resource/train_features.csv'

# 仅仅通过0 和-1 进行填充缺失值
train_feature_filna_path ='Resource/train_features_filna.csv'

less_label_data='Test/features/label_data.csv'
less_feature_data='Test/features/feature.csv'