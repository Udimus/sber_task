"""
Tools to load and preprocess data
"""
import logging

from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
CAT_FEATURES = [
    'feat_1',
    'feat_2',
    'feat_4',
    'feat_11',
    'feat_15',
    'feat_20',
    'feat_27',
    'feat_30',
    'feat_38',
    'feat_39',
    'feat_43',
    'feat_48',
    'feat_50',
    'feat_53',
    'feat_54',
    'feat_57',
    'feat_58',
    'feat_61',
    'feat_62',
    'feat_66',
    'feat_67',
    'feat_69',
    'feat_72',
    'feat_73',
    'feat_77',
    'feat_78',
    'feat_81',
    'feat_88',
    'feat_89',
    'feat_92',
    'feat_96',
    'feat_98',
    'feat_103',
    'feat_104',
    'feat_106',
    'feat_112',
    'feat_113',
    'feat_114',
    'feat_115',
    'feat_121',
    'feat_123',
    'feat_126',
    'feat_128',
    'feat_129',
]
NUM_FEATURES = [
    'feat_5',
    'feat_9',
    'feat_12',
    'feat_22',
    'feat_25',
    'feat_37',
    'feat_55',
    'feat_56',
    'feat_59',
    'feat_60',
    'feat_86',
    'feat_95',
    'feat_119',
    'feat_125',
]
BIN_FEATURES = [
    'feat_0',
    'feat_3',
    'feat_6',
    'feat_7',
    'feat_8',
    'feat_10',
    'feat_13',
    'feat_14',
    'feat_16',
    'feat_17',
    'feat_18',
    'feat_19',
    'feat_21',
    'feat_23',
    'feat_24',
    'feat_26',
    'feat_28',
    'feat_29',
    'feat_31',
    'feat_32',
    'feat_33',
    'feat_34',
    'feat_35',
    'feat_36',
    'feat_40',
    'feat_41',
    'feat_42',
    'feat_44',
    'feat_45',
    'feat_46',
    'feat_47',
    'feat_49',
    'feat_51',
    'feat_52',
    'feat_63',
    'feat_64',
    'feat_65',
    'feat_68',
    'feat_70',
    'feat_71',
    'feat_74',
    'feat_75',
    'feat_76',
    'feat_79',
    'feat_80',
    'feat_82',
    'feat_83',
    'feat_84',
    'feat_85',
    'feat_87',
    'feat_90',
    'feat_91',
    'feat_93',
    'feat_94',
    'feat_97',
    'feat_99',
    'feat_100',
    'feat_101',
    'feat_102',
    'feat_105',
    'feat_107',
    'feat_108',
    'feat_109',
    'feat_110',
    'feat_111',
    'feat_116',
    'feat_117',
    'feat_118',
    'feat_120',
    'feat_122',
    'feat_124',
    'feat_127',
]
EXP = 0.00001


class BinFeaturesTransformer:
    """
    Preprocess binary features as categorical or numeric
    """
    def __init__(
            self,
            num_features=NUM_FEATURES,
            cat_features=CAT_FEATURES,
            bin_features=BIN_FEATURES,
            bin_as_numeric=False,
    ):
        self._cat_features = cat_features
        self._num_features = num_features
        self._bin_features = bin_features
        self._bin_as_numeric = bin_as_numeric
        self._encoders = dict()

    def fit(self, df):
        logger.debug('Fitting Bin transformer...')
        if self._bin_as_numeric:
            for col in self._bin_features:
                encoder = LabelEncoder()
                encoder.fit(df[col])
                self._encoders[col] = encoder
        return self

    def transform(self, df):
        transformed_df = df[
            self._cat_features
            + self._num_features
            + self._bin_features
        ].copy()
        logger.debug('Applying Bin transformer...')
        if self._bin_as_numeric:
            for col in self._bin_features:
                encoder = self._encoders[col]
                transformed_df[col] = encoder.transform(transformed_df[col])
        return transformed_df

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def get_features(self):
        if self._bin_as_numeric:
            return {
                'num_features': self._num_features + self._bin_features,
                'cat_features': self._cat_features,
            }

        return {
            'num_features': self._num_features,
            'cat_features': self._cat_features + self._bin_features,
        }


class CatFeaturesTransformer:
    """
    Convert categorical features to numeric using target.
    """
    def __init__(
            self,
            stat_type='mean',
            expanding=False,
            folds_number=5,
            alpha=0,
    ):
        """
        :param stat_type: statistic type to replace categorical value.
            Now only 'mean' is implemented.
        :param expanding:
            If True, we place at each place in train dataset
            statistic of target from previous n-1 rows.
            If False, for train we use k-folds for train dataset,
            for test we use full train statistic.
        :param folds_number:
            Number of folds. Only if expanding is False.
        :param alpha:
            Smoothing coefficient.
            For example, we want to replace 'value' for 'cat'.
            If sub_df = df[df['cat']=='value'], n_rows = len(sub_df),
            than
            'cat_value' -> (stat(sub_df[target]) * n_rows
                            + stat(df) * alpha) / (n_rows + alpha)
            alpha==0 means no smoothing.
        """
        self._alpha = alpha
        self._folds_number = folds_number
        self._stat_type = stat_type
        self._use_expanding = expanding
        self._encodings = dict()
        self._global_mean = None

    def fit(self, df, cat_features=CAT_FEATURES, target='target'):
        logger.debug('Fitting Cat Transformer...')
        self._global_mean = df[target].mean()
        logger.debug('global mean: %.4f', self._global_mean)
        for col in cat_features:
            cat_sum = df.groupby(col)[target].sum()
            cat_count = df.groupby(col)[target].count()
            cat_mean = ((cat_sum + self._global_mean * self._alpha)
                        / (cat_count + self._alpha))
            self._encodings[col] = cat_mean.to_dict()

    def transform(self, df):
        df = df.copy()
        for col in self._encodings:
            df[col] = (
                df[col]
                .map(self._encodings[col])
                .fillna(self._global_mean)
                .values
            )
        return df

    def fit_transform(self, df, cat_features=CAT_FEATURES, target='target'):
        self._global_mean = df[target].mean()
        df = df.copy()
        if self._use_expanding:
            for col in cat_features:
                cumsum = df.groupby(col)[target].cumsum() - df[target]
                cumcount = df.groupby(col).cumcount()
                df[col] = ((cumsum + self._global_mean * self._alpha)
                           / (cumcount + self._alpha)).fillna(0)
        return df
