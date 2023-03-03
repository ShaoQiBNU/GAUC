from sklearn.metrics import roc_auc_score
from pyspark.sql import SparkSession


def prefilter(df):

    df_rdd = df.rdd.map(lambda x: (x[0], (x[1], x[2]))).groupByKey()
    df_rdd = df_rdd.filter(lambda x: len(x[1]) > 1)
    return df_rdd


def reversePairs(data):
    ans = 0
    prefix = []
    for n in data:
        left, right = 0, len(prefix)
        while left < right:
            mid = (left + right) // 2
            if n >= prefix[mid]:
                left = mid + 1
            else:
                right = mid
        ans += len(prefix) - left
        prefix[left:left] = [n]
    return ans


def cal_auc_custom(f):
    n = len(f)
    label, score = [], []
    for i in f:
        label.append(i[0])
        score.append(i[1])
    try:
        auc = roc_auc_score(label, score)
    except ValueError:
        auc = 0
    return n, auc


def cal_pos_neg_custom(f):
    n = len(f)
    try:
        rank_pos = [values2 for values1, values2 in sorted(f, key=lambda x: x[0], reverse=True)]
        pos_num = reversePairs(rank_pos)
    except ValueError:
        pos_num = 0

    auc = pos_num / (n * (n - 1) / 2)

    return n, auc


def cal_gauc(df, is_spearmanr=False):

    df_rdd = prefilter(df)
    if is_spearmanr:
        df_rdd = df_rdd.mapValues(lambda x: cal_pos_neg_custom(x))
    else:
        df_rdd = df_rdd.mapValues(lambda x: cal_auc_custom(x))

    result = df_rdd.map(lambda x: x[1]).collect()
    sum_impression = sum([0 if i[1] == 0 else i[0] for i in result])
    weight_auc = sum([i[0] * i[1] for i in result])

    return weight_auc / sum_impression


def cal_label_gauc(df, label_columns, predict, key='user_id', label_spearmanr=['staytime']):
    label_gauc = {}
    for label in label_columns:
        is_spearmanr = True if label in label_spearmanr else False
        label_gauc[label] = cal_gauc(df[[key, label, predict]], is_spearmanr=is_spearmanr)

    return label_gauc


def read_dataset(path):
    """
    :param path: hdfs path
    :return: dataframe: feature label
    """
    
    # json格式数据读取
    spark = SparkSession.builder \
        .appName("Python Spark SQL basic example") \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()

    df = spark.read.json(path)
    
    
    '''
    # hive sql读取
    spark_session = SparkSession.builder\
            .enableHiveSupport()\
            .config("hive.exec.dynamic.partition", "true")\
            .config("hive.exec.dynamic.partition.mode", "nonstrict")\
            .getOrCreate()
    df = []
    try:
        df = spark_session.sql(sql)
    except Exception as e:
        print('exec sql on spark failed: sql={}, exception={}'.format(sql, str(e)))
    '''
    df = df.fillna(0)

    print("*" * 200)
    print("dataset description:")
    df.describe().show()

    return df


def main():

    label_spearmanr = ['staytime']
    group_column = 'user_id'
    base_score_column = 'video_multi_adjust_score'
    labels = ['finish', 'follow', 'staytime']

    df = read_dataset("hdfs://R2/projects/szci_video_vec/hdfs/dev/qi.shao/2022-10-19/")
    label_gauc = cal_label_gauc(df, labels, base_score_column, key=group_column, label_spearmanr=label_spearmanr)
    print("%" * 200)
    print("label gauc: ")
    print(label_gauc)


if __name__ == "__main__":
    main()
